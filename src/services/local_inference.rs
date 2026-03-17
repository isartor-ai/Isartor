#![allow(dead_code)]
#![allow(clippy::trim_split_whitespace)]
// =============================================================================
// Embedded Classifier — Rust-native intent classification using candle.
//
// Loads a Qwen2-1.5B-Instruct quantised GGUF model directly into the
// process, eliminating the need for an external llama.cpp sidecar.
// Inference runs on CPU via candle's quantised model support.
//
// Architecture Decision:
//   Instead of calling an external HTTP process, we embed the model
//   weights inside the Rust binary's heap. This reduces operational
//   complexity (one fewer container), removes network latency for
//   classification calls, and enables fine-grained Rust-level
//   observability over every inference step.
//
// Model:
//   Qwen2-1.5B-Instruct quantised to Q4_K_M (~1.1 GB).
//   The GGUF format is loaded via candle's `quantized_qwen2::ModelWeights`.
//
// Thread Safety:
//   `ModelWeights::forward` requires `&mut self`, so we wrap the model
//   in a `tokio::sync::Mutex` to serialise inference calls. Each call
//   is dispatched to `spawn_blocking` to avoid starving the async
//   runtime with CPU-bound tensor operations.
// =============================================================================

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::Repo;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

// ═════════════════════════════════════════════════════════════════════
// Configuration
// ═════════════════════════════════════════════════════════════════════

/// Configuration for the embedded classifier.
#[derive(Debug, Clone)]
pub struct EmbeddedClassifierConfig {
    /// Hugging Face repository ID hosting the GGUF model.
    /// Default: `"Qwen/Qwen2-1.5B-Instruct-GGUF"`
    pub repo_id: String,

    /// Filename of the GGUF model inside the repository.
    /// Default: `"qwen2-1_5b-instruct-q4_k_m.gguf"`
    pub gguf_filename: String,

    /// Optional local path to a pre-downloaded GGUF model file.
    /// When set (e.g. via `ISARTOR__EMBEDDED__MODEL_PATH`), the firewall
    /// skips the Hugging Face download and loads directly from this path.
    pub model_path: Option<String>,

    /// Maximum number of tokens to generate for classification responses.
    /// Classification labels are short, so 20 is usually sufficient.
    pub max_classify_tokens: usize,

    /// Maximum number of tokens to generate for free-form simple task execution.
    pub max_generate_tokens: usize,

    /// Temperature for sampling (0.0 = greedy, >0 = stochastic).
    /// Classification should use 0.0 for deterministic output.
    pub temperature: f64,

    /// Repetition penalty (1.0 = no penalty). Helps avoid degenerate loops.
    pub repetition_penalty: f32,
}

impl Default for EmbeddedClassifierConfig {
    fn default() -> Self {
        Self {
            repo_id: "Qwen/Qwen2-1.5B-Instruct-GGUF".into(),
            gguf_filename: "qwen2-1_5b-instruct-q4_k_m.gguf".into(),
            model_path: None,
            max_classify_tokens: 20,
            max_generate_tokens: 256,
            temperature: 0.0,
            repetition_penalty: 1.1,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Prompt Templates — Qwen2 ChatML format
// ═════════════════════════════════════════════════════════════════════

/// System prompt instructing the model to perform intent classification.
const CLASSIFY_SYSTEM_PROMPT: &str = "\
You are a request classifier for an AI Prompt Firewall. Analyse the user's prompt and \
classify it into EXACTLY ONE of these categories:\n\n\
- SIMPLE — Greetings, basic factual questions, short answers, simple math.\n\
- COMPLEX — Deep reasoning, multi-step analysis, creative writing, long explanations.\n\
- RAG — Questions that need external documents, knowledge base lookups, or citations.\n\
- CODEGEN — Code generation, debugging, implementation, programming tasks.\n\n\
Reply with EXACTLY this format (no other text):\n\
LABEL: <one of SIMPLE|COMPLEX|RAG|CODEGEN>\n\
CONFIDENCE: <a number between 0.0 and 1.0>";

/// Format a classification prompt using the Qwen2 ChatML template.
///
/// Qwen2 ChatML format:
/// ```text
/// <|im_start|>system
/// {system}<|im_end|>
/// <|im_start|>user
/// {prompt}<|im_end|>
/// <|im_start|>assistant
/// ```
pub fn format_classify_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{CLASSIFY_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Format a simple task execution prompt using the Qwen2 ChatML template.
pub fn format_simple_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

// ═════════════════════════════════════════════════════════════════════
// Response Parsing
// ═════════════════════════════════════════════════════════════════════

/// Parse the raw model output into an intent label and confidence score.
///
/// Expected format:
/// ```text
/// LABEL: SIMPLE
/// CONFIDENCE: 0.95
/// ```
///
/// Falls back to ("COMPLEX", 0.0) if parsing fails — this is the safest
/// default since Complex routes to the full pipeline.
pub fn parse_classify_response(raw: &str) -> (String, f64) {
    let upper = raw.to_uppercase();

    // Extract label.
    let label = if let Some(rest) = upper.split("LABEL:").nth(1) {
        let token = rest.trim().split_whitespace().next().unwrap_or("COMPLEX");
        match token {
            "SIMPLE" => "SIMPLE",
            "COMPLEX" => "COMPLEX",
            "RAG" => "RAG",
            "CODEGEN" => "CODEGEN",
            _ => "COMPLEX",
        }
        .to_string()
    } else {
        // Fallback: try to find the label keyword anywhere in the response.
        if upper.contains("SIMPLE") {
            "SIMPLE".into()
        } else if upper.contains("CODEGEN") || upper.contains("CODE") {
            "CODEGEN".into()
        } else if upper.contains("RAG") {
            "RAG".into()
        } else {
            "COMPLEX".into()
        }
    };

    // Extract confidence.
    let confidence = upper
        .split("CONFIDENCE:")
        .nth(1)
        .and_then(|rest| rest.trim().split_whitespace().next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);

    (label, confidence)
}

// ═════════════════════════════════════════════════════════════════════
// EmbeddedClassifier
// ═════════════════════════════════════════════════════════════════════

/// A Rust-native intent classifier and simple-task executor powered by
/// a quantised Qwen2-1.5B-Instruct model loaded via the candle framework.
///
/// The model is loaded once at startup and held in memory. Inference
/// calls are serialised via a `Mutex<ModelWeights>` and dispatched to
/// `spawn_blocking` to avoid blocking the async Tokio runtime.
pub struct EmbeddedClassifier {
    /// The loaded tokenizer for the model.
    tokenizer: Arc<Tokenizer>,

    /// Quantised model weights (requires `&mut self` for forward pass).
    /// Wrapped in a Mutex because candle's `ModelWeights::forward`
    /// takes `&mut self` to update the internal KV cache masks.
    model: Arc<Mutex<ModelWeights>>,

    /// Compute device (CPU for now — maximum VPS compatibility).
    device: Device,

    /// Runtime configuration.
    config: EmbeddedClassifierConfig,

    /// End-of-turn token ID for Qwen2 (`<|im_end|>`).
    eot_token_id: u32,
}

impl EmbeddedClassifier {
    /// Create a new EmbeddedClassifier by downloading and loading the
    /// model from Hugging Face.
    ///
    /// This is an async function because `hf_hub::Api` performs
    /// network I/O to locate and cache model files. The actual model
    /// weight loading (CPU-bound) is done inside `spawn_blocking`.
    pub async fn new(cfg: EmbeddedClassifierConfig) -> Result<Self> {
        let load_start = Instant::now();

        tracing::info!(
            "EmbeddedClassifier: initialising, repo: {}, gguf: {}, model_path: {:?}",
            cfg.repo_id,
            cfg.gguf_filename,
            cfg.model_path
        );

        // ── Step 1: Locate model files ───────────────────────────
        let cache_dir = crate::hf::writable_hf_hub_cache_dir()
            .context("failed to determine a writable hf-hub cache directory")?;
        let api = ApiBuilder::from_env()
            .with_progress(false)
            .with_cache_dir(cache_dir.clone())
            .build()
            .with_context(|| {
                format!(
                    "failed to create Hugging Face API client (cache_dir: {})",
                    cache_dir.display()
                )
            })?;

        // If a local model_path is configured (e.g. baked into Docker image),
        // use it directly; otherwise download via hf-hub.
        let model_path = if let Some(ref local) = cfg.model_path {
            let p = PathBuf::from(local);
            if p.exists() {
                tracing::info!(
                    "EmbeddedClassifier: using local model file: {}",
                    p.display()
                );
                p
            } else {
                tracing::warn!(
                    "EmbeddedClassifier: configured model_path {} not found, falling back to hf-hub download",
                    p.display()
                );
                let repo = api.repo(Repo::model(cfg.repo_id.clone()));
                repo.get(&cfg.gguf_filename)
                    .await
                    .context("failed to download GGUF model file")?
            }
        } else {
            let repo = api.repo(Repo::model(cfg.repo_id.clone()));
            repo.get(&cfg.gguf_filename)
                .await
                .context("failed to download GGUF model file")?
        };

        // Download tokenizer — tries the GGUF repo first (often includes
        // tokenizer.json), then falls back to the canonical Google repo.
        let tokenizer_path = Self::resolve_tokenizer_path(&api, &cfg.repo_id).await?;

        tracing::info!(
            "EmbeddedClassifier: model files located, model_path: {}, tokenizer_path: {}, download_ms: {}",
            model_path.display(), tokenizer_path.display(), load_start.elapsed().as_millis()
        );

        // ── Step 2: Load model (CPU-bound) ───────────────────────
        let device = Device::Cpu;
        let device_clone = device.clone();
        let model_path_clone = model_path.clone();

        let model = tokio::task::spawn_blocking(move || {
            Self::load_model_weights(&model_path_clone, &device_clone)
        })
        .await
        .context("model loading task panicked")?
        .context("failed to load quantised model weights")?;

        // ── Step 3: Load tokenizer ───────────────────────────────
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        // Resolve the end-of-turn token ID.
        let eot_token_id = tokenizer.token_to_id("<|im_end|>").unwrap_or_else(|| {
            tracing::warn!("EmbeddedClassifier: <|im_end|> token not found, using EOS fallback");
            // Qwen2 uses <|endoftext|> as EOS.
            tokenizer.token_to_id("<|endoftext|>").unwrap_or(1)
        });

        let total_load_ms = load_start.elapsed().as_millis();
        tracing::info!(
            "EmbeddedClassifier: model loaded successfully on CPU, total_load_ms: {}, eot_token_id: {}",
            total_load_ms, eot_token_id
        );

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            model: Arc::new(Mutex::new(model)),
            device,
            config: cfg,
            eot_token_id,
        })
    }

    /// Resolve the tokenizer.json path — tries the quantised GGUF repo first,
    /// then the canonical Qwen2 repo, then a local fallback.
    async fn resolve_tokenizer_path(api: &Api, gguf_repo_id: &str) -> Result<PathBuf> {
        // Repos to try, in order.  The GGUF repo is tried first because some
        // re-uploaders include tokenizer.json.
        let repos_to_try = vec![
            gguf_repo_id.to_string(),
            "Qwen/Qwen2-1.5B-Instruct".to_string(),
        ];

        for repo_id in &repos_to_try {
            let repo = api.repo(Repo::model(repo_id.clone()));
            match repo.get("tokenizer.json").await {
                Ok(path) => {
                    tracing::info!("Tokenizer found in repo: {}", repo_id);
                    return Ok(path);
                }
                Err(e) => {
                    tracing::debug!(
                        "tokenizer.json not found in repo {}: {}, trying next",
                        repo_id,
                        e
                    );
                }
            }
        }

        // Last resort: look in the current working directory.
        let local_path = PathBuf::from("tokenizer.json");
        if local_path.exists() {
            tracing::info!("Using local tokenizer.json from working directory");
            return Ok(local_path);
        }

        anyhow::bail!(
            "Could not locate tokenizer.json. Tried repos {:?} \
             and the working directory. Set HF_TOKEN for gated models or place tokenizer.json locally.",
            repos_to_try
        )
    }

    /// Load quantised GGUF model weights (synchronous, CPU-bound).
    fn load_model_weights(model_path: &PathBuf, device: &Device) -> Result<ModelWeights> {
        let load_start = Instant::now();

        tracing::info!(
            "EmbeddedClassifier: loading GGUF weights into memory, path: {}",
            model_path.display()
        );

        let mut file = std::fs::File::open(model_path).context("failed to open GGUF model file")?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("failed to parse GGUF content: {e}"))?;

        tracing::debug!(
            "GGUF file parsed, tensor_infos: {}, metadata_entries: {}",
            content.tensor_infos.len(),
            content.metadata.len()
        );

        let weights = ModelWeights::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("failed to build ModelWeights from GGUF: {e}"))?;

        tracing::info!(
            "EmbeddedClassifier: GGUF weights loaded, load_ms: {}",
            load_start.elapsed().as_millis()
        );

        Ok(weights)
    }

    /// Perform intent classification on a user prompt.
    ///
    /// Formats the prompt with the Qwen2 ChatML template, runs a
    /// greedy token generation loop, and parses the structured output.
    ///
    /// This method is safe to call from async code — it dispatches the
    /// CPU-bound inference to `spawn_blocking`.
    pub async fn classify(&self, prompt: &str) -> Result<(String, f64)> {
        let formatted = format_classify_prompt(prompt);
        let raw = self
            .generate(&formatted, self.config.max_classify_tokens)
            .await?;

        tracing::debug!("EmbeddedClassifier: raw classification output: {}", raw);

        Ok(parse_classify_response(&raw))
    }

    /// Execute a simple task — generate a free-form response.
    ///
    /// Used when the intent classifier determines the task is simple
    /// enough to handle locally without the external LLM.
    pub async fn execute(&self, prompt: &str) -> Result<String> {
        let formatted = format_simple_prompt(prompt);
        self.generate(&formatted, self.config.max_generate_tokens)
            .await
    }

    /// Core generation loop: tokenise → forward pass → greedy sample → decode.
    ///
    /// Dispatches to `spawn_blocking` to avoid blocking the Tokio runtime.
    async fn generate(&self, formatted_prompt: &str, max_tokens: usize) -> Result<String> {
        let inference_start = Instant::now();

        // Tokenise the prompt.
        let encoding = self
            .tokenizer
            .encode(formatted_prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encoding failed: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        tracing::debug!(
            "EmbeddedClassifier: starting generation, prompt_tokens: {}, max_tokens: {}",
            prompt_len,
            max_tokens
        );

        // Clone Arcs for the blocking closure.
        let model = self.model.clone();
        let device = self.device.clone();
        let eot = self.eot_token_id;
        let rep_penalty = self.config.repetition_penalty;
        let temperature = self.config.temperature;

        // Run inference on a blocking thread.
        let generated_tokens: Vec<u32> = tokio::task::spawn_blocking(move || {
            // We acquire the mutex synchronously inside the blocking thread.
            // This is safe because we're NOT on the async runtime.
            let mut model_guard = model.blocking_lock();
            Self::generate_tokens(
                &mut model_guard,
                &device,
                &prompt_tokens,
                max_tokens,
                eot,
                temperature,
                rep_penalty,
            )
        })
        .await
        .context("generation task panicked")?
        .context("token generation failed")?;

        let output = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

        let inference_ms = inference_start.elapsed().as_millis();
        let tokens_generated = generated_tokens.len();
        let tokens_per_sec = if inference_ms > 0 {
            (tokens_generated as f64 / inference_ms as f64) * 1000.0
        } else {
            0.0
        };

        tracing::info!(
            "EmbeddedClassifier: generation complete, prompt_tokens: {}, tokens_generated: {}, inference_ms: {}, tokens_per_sec: {:.1}",
            prompt_len, tokens_generated, inference_ms, tokens_per_sec
        );

        Ok(output)
    }

    /// Synchronous greedy token generation loop.
    ///
    /// Runs entirely on CPU. Generates tokens one at a time using the
    /// model's forward pass and argmax sampling.
    fn generate_tokens(
        model: &mut ModelWeights,
        device: &Device,
        prompt_tokens: &[u32],
        max_tokens: usize,
        eot_token_id: u32,
        temperature: f64,
        repetition_penalty: f32,
    ) -> Result<Vec<u32>> {
        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

        // ── Prefill: process the entire prompt in one forward pass ──
        let prompt_tensor = Tensor::new(prompt_tokens, device)
            .map_err(|e| anyhow::anyhow!("tensor creation failed: {e}"))?
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("unsqueeze failed: {e}"))?;

        let logits = model
            .forward(&prompt_tensor, 0)
            .map_err(|e| anyhow::anyhow!("forward pass (prefill) failed: {e}"))?;

        let mut next_token =
            Self::sample_token(&logits, &all_tokens, temperature, repetition_penalty)?;

        generated.push(next_token);
        all_tokens.push(next_token);

        if next_token == eot_token_id {
            return Ok(generated);
        }

        // ── Decode: generate one token at a time ────────────────────
        for i in 0..max_tokens.saturating_sub(1) {
            let input = Tensor::new(&[next_token], device)
                .map_err(|e| anyhow::anyhow!("tensor creation failed: {e}"))?
                .unsqueeze(0)
                .map_err(|e| anyhow::anyhow!("unsqueeze failed: {e}"))?;

            let index_pos = prompt_tokens.len() + i + 1;

            let logits = model
                .forward(&input, index_pos)
                .map_err(|e| anyhow::anyhow!("forward pass (decode step {i}) failed: {e}"))?;

            next_token = Self::sample_token(&logits, &all_tokens, temperature, repetition_penalty)?;

            if next_token == eot_token_id {
                break;
            }

            generated.push(next_token);
            all_tokens.push(next_token);
        }

        Ok(generated)
    }

    /// Sample the next token from logits using greedy (argmax) or
    /// temperature-scaled sampling with repetition penalty.
    fn sample_token(
        logits: &Tensor,
        all_tokens: &[u32],
        temperature: f64,
        repetition_penalty: f32,
    ) -> Result<u32> {
        let logits = logits
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("squeeze failed: {e}"))?;

        // Apply repetition penalty.
        let mut logits_vec: Vec<f32> = logits
            .to_vec1()
            .map_err(|e| anyhow::anyhow!("logits to_vec1 failed: {e}"))?;

        if repetition_penalty != 1.0 {
            for &token_id in all_tokens {
                if let Some(logit) = logits_vec.get_mut(token_id as usize) {
                    if *logit > 0.0 {
                        *logit /= repetition_penalty;
                    } else {
                        *logit *= repetition_penalty;
                    }
                }
            }
        }

        if temperature <= 0.0 || temperature < 1e-7 {
            // Greedy: pick the argmax.
            let next_token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| anyhow::anyhow!("empty logits vector"))?;
            Ok(next_token)
        } else {
            // Temperature-scaled softmax sampling.
            let temp = temperature as f32;
            let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let exp_sum: f32 = logits_vec
                .iter()
                .map(|&l| ((l - max_logit) / temp).exp())
                .sum();

            let probs: Vec<f32> = logits_vec
                .iter()
                .map(|&l| ((l - max_logit) / temp).exp() / exp_sum)
                .collect();

            // Weighted random selection.
            let random_val: f32 = {
                // Simple deterministic fallback using token history as seed.
                // In production, use a proper RNG.
                let seed = all_tokens.iter().map(|&t| t as u64).sum::<u64>();
                let hash = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (hash as f32 / u64::MAX as f32).abs()
            };

            let mut cumulative = 0.0f32;
            for (idx, &prob) in probs.iter().enumerate() {
                cumulative += prob;
                if cumulative >= random_val {
                    return Ok(idx as u32);
                }
            }

            // Fallback: last token.
            Ok((probs.len() - 1) as u32)
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Prompt Formatting ────────────────────────────────────────

    #[test]
    fn classify_prompt_contains_chatml_template() {
        let prompt = "What is 2+2?";
        let formatted = format_classify_prompt(prompt);
        assert!(formatted.contains("<|im_start|>system\n"));
        assert!(formatted.contains("What is 2+2?"));
        assert!(formatted.contains(CLASSIFY_SYSTEM_PROMPT));
        assert!(formatted.ends_with("<|im_start|>assistant\n"));
        assert!(formatted.contains("<|im_end|>"));
    }

    #[test]
    fn simple_prompt_contains_chatml_template() {
        let prompt = "Hello, world!";
        let formatted = format_simple_prompt(prompt);
        assert!(formatted.contains("<|im_start|>user\n"));
        assert!(formatted.contains("Hello, world!"));
        assert!(formatted.ends_with("<|im_start|>assistant\n"));
        // Simple prompt should NOT contain the classify system prompt.
        assert!(!formatted.contains("EXACTLY ONE"));
    }

    #[test]
    fn classify_prompt_escapes_special_chars() {
        let prompt = "What is <script>alert('xss')</script>?";
        let formatted = format_classify_prompt(prompt);
        // The prompt should be included verbatim (no HTML escaping needed).
        assert!(formatted.contains("<script>alert('xss')</script>"));
    }

    // ── Response Parsing ─────────────────────────────────────────

    #[test]
    fn parse_label_simple() {
        let (label, confidence) = parse_classify_response("LABEL: SIMPLE\nCONFIDENCE: 0.95");
        assert_eq!(label, "SIMPLE");
        assert!((confidence - 0.95).abs() < 1e-9);
    }

    #[test]
    fn parse_label_complex() {
        let (label, confidence) = parse_classify_response("LABEL: COMPLEX\nCONFIDENCE: 0.80");
        assert_eq!(label, "COMPLEX");
        assert!((confidence - 0.80).abs() < 1e-9);
    }

    #[test]
    fn parse_label_rag() {
        let (label, confidence) = parse_classify_response("LABEL: RAG\nCONFIDENCE: 0.70");
        assert_eq!(label, "RAG");
        assert!((confidence - 0.70).abs() < 1e-9);
    }

    #[test]
    fn parse_label_codegen() {
        let (label, confidence) = parse_classify_response("LABEL: CODEGEN\nCONFIDENCE: 0.88");
        assert_eq!(label, "CODEGEN");
        assert!((confidence - 0.88).abs() < 1e-9);
    }

    #[test]
    fn parse_label_case_insensitive() {
        let (label, _) = parse_classify_response("label: simple\nconfidence: 0.9");
        assert_eq!(label, "SIMPLE");
    }

    #[test]
    fn parse_fallback_keyword_detection_simple() {
        // No "LABEL:" prefix, but "simple" appears in the text.
        let (label, confidence) = parse_classify_response("This is a simple question.");
        assert_eq!(label, "SIMPLE");
        assert_eq!(confidence, 0.0); // No CONFIDENCE: line.
    }

    #[test]
    fn parse_fallback_keyword_detection_codegen() {
        let (label, _) = parse_classify_response("The user wants to write code in Rust");
        assert_eq!(label, "CODEGEN");
    }

    #[test]
    fn parse_fallback_keyword_detection_rag() {
        let (label, _) = parse_classify_response("This is a RAG retrieval question");
        assert_eq!(label, "RAG");
    }

    #[test]
    fn parse_fallback_unknown_defaults_to_complex() {
        let (label, confidence) = parse_classify_response("something unexpected");
        assert_eq!(label, "COMPLEX");
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn parse_confidence_clamped_above_one() {
        let (_, confidence) = parse_classify_response("LABEL: SIMPLE\nCONFIDENCE: 1.5");
        assert!((confidence - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_confidence_clamped_below_zero() {
        let (_, confidence) = parse_classify_response("LABEL: SIMPLE\nCONFIDENCE: -0.5");
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn parse_confidence_missing_defaults_to_zero() {
        let (label, confidence) = parse_classify_response("LABEL: SIMPLE");
        assert_eq!(label, "SIMPLE");
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn parse_empty_response() {
        let (label, confidence) = parse_classify_response("");
        assert_eq!(label, "COMPLEX");
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn parse_multiline_with_extra_text() {
        let raw = "Thinking about this...\nLABEL: RAG\nCONFIDENCE: 0.85\nSome trailing text.";
        let (label, confidence) = parse_classify_response(raw);
        assert_eq!(label, "RAG");
        assert!((confidence - 0.85).abs() < 1e-9);
    }

    // ── EmbeddedClassifierConfig ─────────────────────────────────

    #[test]
    fn default_config_values() {
        let cfg = EmbeddedClassifierConfig::default();
        assert_eq!(cfg.repo_id, "Qwen/Qwen2-1.5B-Instruct-GGUF");
        assert_eq!(cfg.gguf_filename, "qwen2-1_5b-instruct-q4_k_m.gguf");
        assert!(cfg.model_path.is_none());
        assert_eq!(cfg.max_classify_tokens, 20);
        assert_eq!(cfg.max_generate_tokens, 256);
        assert!((cfg.temperature - 0.0).abs() < 1e-9);
        assert!((cfg.repetition_penalty - 1.1).abs() < 1e-6);
    }

    #[test]
    fn config_clone() {
        let cfg = EmbeddedClassifierConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cfg.repo_id, cloned.repo_id);
        assert_eq!(cfg.gguf_filename, cloned.gguf_filename);
    }

    // ── Sampling Logic ───────────────────────────────────────────

    #[test]
    fn greedy_sampling_picks_max() {
        // Create a small logits tensor: [0.1, 0.5, 0.3, 0.9]
        // Greedy should pick index 3.
        let device = Device::Cpu;
        let logits = Tensor::new(&[0.1f32, 0.5, 0.3, 0.9], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let token = EmbeddedClassifier::sample_token(&logits, &[], 0.0, 1.0).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn greedy_sampling_with_repetition_penalty() {
        // Logits: [0.1, 0.5, 0.3, 0.9]
        // Token 3 already used → penalty reduces its logit.
        // With penalty 2.0: logit[3] = 0.9 / 2.0 = 0.45 < 0.5
        // So token 1 (logit 0.5) should be selected instead.
        let device = Device::Cpu;
        let logits = Tensor::new(&[0.1f32, 0.5, 0.3, 0.9], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let token = EmbeddedClassifier::sample_token(&logits, &[3], 0.0, 2.0).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn repetition_penalty_1_0_is_noop() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[0.1f32, 0.5, 0.3, 0.9], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        // With penalty 1.0, token 3 should still be selected.
        let token = EmbeddedClassifier::sample_token(&logits, &[3], 0.0, 1.0).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn negative_logit_repetition_penalty() {
        // Logits: [-0.5, -0.1, -0.8, 0.2]
        // Without penalty: token 3 (0.2) wins.
        // Token 3 with penalty 2.0: 0.2 / 2.0 = 0.1 > -0.1
        // Token 3 still wins (0.1 > -0.1).
        //
        // Token 1 with penalty: already negative, so multiplied: -0.1 * 2.0 = -0.2
        // So token 3 (0.1) still wins.
        let device = Device::Cpu;
        let logits = Tensor::new(&[-0.5f32, -0.1, -0.8, 0.2], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let token = EmbeddedClassifier::sample_token(&logits, &[3], 0.0, 2.0).unwrap();
        assert_eq!(token, 3); // still highest after penalty
    }

    #[test]
    fn temperature_sampling_returns_valid_token() {
        // With temperature > 0, the result should be a valid token index.
        let device = Device::Cpu;
        let logits = Tensor::new(&[0.1f32, 0.5, 0.3, 0.9], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let token = EmbeddedClassifier::sample_token(&logits, &[1, 2], 0.8, 1.0).unwrap();
        assert!(token < 4);
    }
}

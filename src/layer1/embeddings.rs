use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::Api;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Pure-Rust sentence embedder powered by candle + all-MiniLM-L6-v2.
///
/// Loads model weights from Hugging Face Hub on first use and runs BERT
/// inference entirely in-process — zero C/C++ dependencies, seamless
/// cross-compilation to any `rustc` target.
///
/// The inner `Mutex` is required because `BertModel::forward()` borrows
/// `&mut self` for the KV cache. We use `std::sync::Mutex` (not tokio)
/// because inference is CPU-bound and always called from `spawn_blocking`.
pub struct TextEmbedder {
    model: Mutex<BertModel>,
    tokenizer: Tokenizer,
    device: Device,
}

impl TextEmbedder {
    /// Downloads (or loads from cache) the `sentence-transformers/all-MiniLM-L6-v2`
    /// model and tokenizer from Hugging Face Hub, then builds the BertModel on CPU.
    ///
    /// This function blocks during startup while downloading the model files (~90 MB total).
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        let repo_id = "sentence-transformers/all-MiniLM-L6-v2";

        // Download model artifacts via hf-hub sync API
        let api = Api::new().context("Failed to initialise Hugging Face Hub API")?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;

        // Load configuration
        let config_bytes = std::fs::read(&config_path).context("Failed to read config.json")?;
        let config: BertConfig = serde_json::from_slice(&config_bytes)
            .context("Failed to parse BertConfig from config.json")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Load model weights from safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
                .context("Failed to mmap model.safetensors")?
        };
        let model = BertModel::load(vb, &config)
            .context("Failed to load BertModel from safetensors weights")?;

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
        })
    }

    /// Generates a 384-dimensional embedding vector for a single prompt string.
    ///
    /// Pipeline: tokenize → BERT forward pass → mean pooling (with attention mask) → L2 normalise.
    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenisation failed: {e}"))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let token_type_ids = encoding.get_type_ids();
        let seq_len = input_ids.len();

        // Build tensors [1, seq_len]
        let input_ids_t = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask_t = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;
        let token_type_ids_t = Tensor::new(token_type_ids, &self.device)?.unsqueeze(0)?;

        // Forward pass through BERT
        let model = self
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("embedder lock poisoned: {e}"))?;
        let output = model.forward(&input_ids_t, &token_type_ids_t, Some(&attention_mask_t))?;
        // output shape: [1, seq_len, hidden_size]

        // Mean pooling: mask padding tokens, average over sequence dimension
        let mask = attention_mask_t
            .to_dtype(candle_core::DType::F32)?
            .unsqueeze(2)?
            .broadcast_as(output.shape())?;
        let masked = (output * mask)?;
        let summed = masked.sum(1)?; // [1, hidden_size]
        let count = Tensor::new(&[seq_len as f32], &self.device)?
            .unsqueeze(0)?
            .broadcast_as(summed.shape())?;
        let mean_pooled = (summed / count)?; // [1, hidden_size]

        // L2 normalise
        let norm = mean_pooled
            .sqr()?
            .sum(1)?
            .sqrt()?
            .unsqueeze(1)?
            .broadcast_as(mean_pooled.shape())?;
        let normalised = (mean_pooled / norm)?;

        // Extract Vec<f32>
        let embedding: Vec<f32> = normalised.squeeze(0)?.to_vec1()?;
        Ok(embedding)
    }
}

/// Helper function to calculate the cosine similarity between two slice references.
/// Formula: (A dot B) / (||A|| * ||B||)
#[allow(dead_code)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (val_a, val_b) in a.iter().zip(b.iter()) {
        dot_product += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

/// Returns a shared singleton `TextEmbedder` for use in tests.
/// Avoids model re-download contention when many tests run in parallel.
#[cfg(test)]
pub fn shared_test_embedder() -> std::sync::Arc<TextEmbedder> {
    use std::sync::{Arc, LazyLock};
    static EMBEDDER: LazyLock<Arc<TextEmbedder>> =
        LazyLock::new(|| Arc::new(TextEmbedder::new().expect("shared test TextEmbedder")));
    EMBEDDER.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_embedder_flow() {
        // Use the shared singleton to avoid redundant model downloads
        let embedder = shared_test_embedder();

        let text1 = "Where is the pricing page?";
        let text2 = "How much does it cost?";
        let text3 = "I like eating apples.";

        // Generate embeddings
        let emb1 = embedder
            .generate_embedding(text1)
            .expect("Failed embedding 1");
        let emb2 = embedder
            .generate_embedding(text2)
            .expect("Failed embedding 2");
        let emb3 = embedder
            .generate_embedding(text3)
            .expect("Failed embedding 3");

        // Verify dimensionality (all-MiniLM-L6-v2 produces 384-dim vectors)
        assert_eq!(emb1.len(), 384, "Expected 384-dimensional embedding");

        // Calculate similarities
        let sim1_2 = cosine_similarity(&emb1, &emb2);
        let sim1_3 = cosine_similarity(&emb1, &emb3);

        println!("Similarity between semantic matches: {}", sim1_2);
        println!("Similarity between unrelated: {}", sim1_3);

        // Assert that the semantic match is reasonably high
        assert!(
            sim1_2 > 0.6,
            "Similarity should be high for semantic matches, got {sim1_2}"
        );
        // Assert that the unrelated text has lower similarity
        assert!(
            sim1_2 > sim1_3,
            "Semantic match should score higher than unrelated"
        );
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0); // Orthogonal

        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < f32::EPSILON); // Identical

        let zero = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(cosine_similarity(&zero, &b), 0.0); // Zero magnitude
    }

    #[test]
    fn test_candle_embedder_prompt_ab_c() {
        let embedder = shared_test_embedder();

        let prompt_a = "What is the pricing?";
        let prompt_b = "How much does it cost?";
        let prompt_c = "Give me a python script for a web scraper.";

        let emb_a = embedder.generate_embedding(prompt_a).expect("embed A");
        let emb_b = embedder.generate_embedding(prompt_b).expect("embed B");
        let emb_c = embedder.generate_embedding(prompt_c).expect("embed C");

        let sim_ab = cosine_similarity(&emb_a, &emb_b);
        let sim_ac = cosine_similarity(&emb_a, &emb_c);

        println!("sim(A, B) = {sim_ab}");
        println!("sim(A, C) = {sim_ac}");

        assert!(
            sim_ab > 0.7,
            "Semantically similar prompts should have > 0.7 similarity, got {sim_ab}"
        );
        assert!(
            sim_ac < 0.3,
            "Unrelated prompts should have < 0.3 similarity, got {sim_ac}"
        );
    }
}

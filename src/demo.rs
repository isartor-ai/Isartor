//! # First-Run Demo Runner
//!
//! Replays a bundled set of prompts against the live L1a/L1b cache layers
//! to demonstrate Isartor's deflection capability.  No external LLM or
//! SLM is required — the demo seeds the caches with synthetic responses
//! and then replays near-duplicate prompts to measure cache hit rates.
//!
//! ## Flow
//!
//! 1. Parse `demo/replay.jsonl` (bundled at compile time).
//! 2. Seed the exact cache with canonical prompts + synthetic responses.
//! 3. Replay all prompts against the cache layers.
//! 4. Collect per-prompt stats (layer hit, latency).
//! 5. Print a summary table to stdout (indicatif).
//! 6. Write `isartor_demo_result.txt` for CI validation.

use std::io::BufRead;
use std::sync::Arc;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

use crate::config::{AppConfig, LlmProvider};
use crate::state::AppState;

// ── Bundled fixture ──────────────────────────────────────────────────

/// The 50-prompt replay fixture, embedded at compile time.
const REPLAY_JSONL: &str = include_str!("../demo/replay.jsonl");

// ── Types ────────────────────────────────────────────────────────────

/// A single prompt from the replay fixture.
#[derive(Debug, Clone, serde::Deserialize)]
struct ReplayPrompt {
    prompt: String,
}

/// Which cache layer resolved a replayed prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoHitLayer {
    /// Layer 1a — exact SHA-256 match.
    ExactCache,
    /// Layer 1b — semantic cosine-similarity match.
    SemanticCache,
    /// No cache hit — would have been forwarded to L2/L3.
    Passthrough,
}

impl DemoHitLayer {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ExactCache => "L1a (exact)",
            Self::SemanticCache => "L1b (semantic)",
            Self::Passthrough => "passthrough",
        }
    }
}

/// Result of replaying a single prompt.
#[derive(Debug)]
pub struct PromptResult {
    pub prompt: String,
    pub layer: DemoHitLayer,
    pub latency: std::time::Duration,
}

/// Aggregate statistics from the demo run.
#[derive(Debug)]
pub struct DemoStats {
    pub total: usize,
    pub exact_hits: usize,
    pub semantic_hits: usize,
    pub passthrough: usize,
    pub deflection_pct: f64,
    pub avg_latency_us: f64,
    pub results: Vec<PromptResult>,
    pub live_showcase: LiveShowcaseResult,
}

/// Optional live upstream showcase for post-install demo runs.
#[derive(Debug)]
pub enum LiveShowcaseResult {
    Success {
        provider: String,
        model: String,
        latency: std::time::Duration,
        response_preview: String,
    },
    Skipped {
        reason: String,
    },
    Failed {
        provider: String,
        model: String,
        error: String,
    },
}

// ── Canonical seeds ──────────────────────────────────────────────────

/// Returns (canonical_prompt, synthetic_response) pairs.
/// These are the "first occurrence" prompts that seed the caches.
fn canonical_seeds() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "What is the capital of France?",
            "The capital of France is Paris.",
        ),
        (
            "Tell me the capital of France",
            "The capital of France is Paris.",
        ),
        (
            "Explain what a REST API is",
            "A REST API is an HTTP interface that exposes resources through predictable URLs and standard verbs such as GET, POST, PUT, and DELETE.",
        ),
        (
            "What is a REST API?",
            "A REST API is an HTTP interface that exposes resources through predictable URLs and standard verbs such as GET, POST, PUT, and DELETE.",
        ),
        (
            "How do I reverse a string in Python?",
            "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```",
        ),
        (
            "How to reverse a string in Python?",
            "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```",
        ),
        (
            "What is the meaning of life?",
            "The meaning of life is a philosophical question explored through many traditions. A popular cultural answer is 42.",
        ),
        (
            "What is machine learning?",
            "Machine learning is a subset of AI where computers learn patterns from data rather than being explicitly programmed.",
        ),
        (
            "Explain machine learning to me",
            "Machine learning is a subset of AI where computers learn patterns from data rather than being explicitly programmed.",
        ),
        (
            "How does photosynthesis work?",
            "Photosynthesis converts sunlight, water, and CO₂ into glucose and oxygen using chlorophyll in plant cells.",
        ),
        (
            "Explain photosynthesis",
            "Photosynthesis converts sunlight, water, and CO₂ into glucose and oxygen using chlorophyll in plant cells.",
        ),
    ]
}

// ── Public API ───────────────────────────────────────────────────────

/// Run the first-run demo against the live `AppState` caches.
///
/// Returns aggregate stats and per-prompt results for display.
pub async fn run_demo(state: &Arc<AppState>) -> anyhow::Result<DemoStats> {
    let live_showcase = run_live_showcase(state).await;

    // 1. Parse the replay fixture.
    let prompts = parse_replay_fixture()?;
    let total = prompts.len();

    // 2. Seed caches with canonical prompts.
    seed_caches(state).await?;

    // 3. Set up progress bar.
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "  {spinner:.green} Replaying [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("█▓░"),
    );

    // 4. Replay all prompts.
    let mut results = Vec::with_capacity(total);
    let mut exact_hits = 0usize;
    let mut semantic_hits = 0usize;
    let mut passthrough = 0usize;

    for rp in &prompts {
        let start = Instant::now();
        let layer = check_cache_layers(state, &rp.prompt).await;
        let latency = start.elapsed();

        match layer {
            DemoHitLayer::ExactCache => exact_hits += 1,
            DemoHitLayer::SemanticCache => semantic_hits += 1,
            DemoHitLayer::Passthrough => passthrough += 1,
        }

        results.push(PromptResult {
            prompt: rp.prompt.clone(),
            layer,
            latency,
        });
        pb.inc(1);
    }
    pb.finish_with_message("Done");

    let deflected = exact_hits + semantic_hits;
    let deflection_pct = if total > 0 {
        (deflected as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    let avg_latency_us = if total > 0 {
        results
            .iter()
            .map(|r| r.latency.as_micros() as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };

    Ok(DemoStats {
        total,
        exact_hits,
        semantic_hits,
        passthrough,
        deflection_pct,
        avg_latency_us,
        results,
        live_showcase,
    })
}

/// Print the demo result table to stdout.
pub fn print_demo_results(stats: &DemoStats) {
    println!();
    println!("  Isartor Demo");
    println!("  ────────────");
    println!(
        "  You just ran the post-install showcase: optional live L3, then local cache replay."
    );
    println!();

    match &stats.live_showcase {
        LiveShowcaseResult::Success {
            provider,
            model,
            latency,
            response_preview,
        } => {
            println!("  Live provider showcase");
            println!("  ─────────────────────");
            println!("  Provider:   {}", provider);
            println!("  Model:      {}", model);
            println!("  Latency:    {:.0} ms", latency.as_secs_f64() * 1_000.0);
            println!("  Preview:    {}", response_preview);
            println!();
        }
        LiveShowcaseResult::Skipped { reason } => {
            println!("  Live provider showcase");
            println!("  ─────────────────────");
            println!("  Skipped:    {}", reason);
            println!();
        }
        LiveShowcaseResult::Failed {
            provider,
            model,
            error,
        } => {
            println!("  Live provider showcase");
            println!("  ─────────────────────");
            println!("  Provider:   {}", provider);
            println!("  Model:      {}", model);
            println!("  Status:     failed");
            println!("  Detail:     {}", error);
            println!();
        }
    }

    println!("  Cache replay");
    println!("  ────────────");
    println!("  ┌──────────────────────────────────────────────────┐");
    println!("  │             Isartor Demo Results                 │");
    println!("  ├──────────────────────────────────────────────────┤");
    println!(
        "  │  Total prompts replayed:    {:>4}                 │",
        stats.total
    );
    println!(
        "  │  L1a exact cache hits:      {:>4}                 │",
        stats.exact_hits
    );
    println!(
        "  │  L1b semantic cache hits:   {:>4}                 │",
        stats.semantic_hits
    );
    println!(
        "  │  Passthrough (→ L2/L3):     {:>4}                 │",
        stats.passthrough
    );
    println!("  ├──────────────────────────────────────────────────┤");
    println!(
        "  │  Deflection rate:       {:>6.1}%                 │",
        stats.deflection_pct
    );
    println!(
        "  │  Avg cache latency:     {:>6.0} µs                │",
        stats.avg_latency_us
    );
    println!("  └──────────────────────────────────────────────────┘");
    println!();
    println!(
        "  Cloud calls avoided locally: {}/{} ({:.1}%)",
        stats.exact_hits + stats.semantic_hits,
        stats.total,
        stats.deflection_pct
    );
    println!("  Next steps:");
    println!("    1. isartor check");
    println!("    2. isartor up --detach");
    println!("    3. isartor connect <tool>");
    println!();
}

/// Write the demo result summary to `isartor_demo_result.txt`.
pub fn write_demo_result_file(stats: &DemoStats) -> std::io::Result<()> {
    use std::io::Write;

    let path = "isartor_demo_result.txt";
    let mut f = std::fs::File::create(path)?;

    writeln!(f, "Isartor First-Run Demo Results")?;
    writeln!(f, "==============================")?;
    writeln!(f, "Total prompts:       {}", stats.total)?;
    writeln!(f, "L1a exact hits:      {}", stats.exact_hits)?;
    writeln!(f, "L1b semantic hits:   {}", stats.semantic_hits)?;
    writeln!(f, "Passthrough:         {}", stats.passthrough)?;
    writeln!(f, "Deflection rate:     {:.1}%", stats.deflection_pct)?;
    writeln!(f, "Avg latency (µs):    {:.0}", stats.avg_latency_us)?;
    writeln!(f)?;
    writeln!(f, "Live provider showcase:")?;
    match &stats.live_showcase {
        LiveShowcaseResult::Success {
            provider,
            model,
            latency,
            response_preview,
        } => {
            writeln!(f, "  status: success")?;
            writeln!(f, "  provider: {}", provider)?;
            writeln!(f, "  model: {}", model)?;
            writeln!(f, "  latency_ms: {:.0}", latency.as_secs_f64() * 1_000.0)?;
            writeln!(f, "  preview: {}", response_preview)?;
        }
        LiveShowcaseResult::Skipped { reason } => {
            writeln!(f, "  status: skipped")?;
            writeln!(f, "  reason: {}", reason)?;
        }
        LiveShowcaseResult::Failed {
            provider,
            model,
            error,
        } => {
            writeln!(f, "  status: failed")?;
            writeln!(f, "  provider: {}", provider)?;
            writeln!(f, "  model: {}", model)?;
            writeln!(f, "  error: {}", error)?;
        }
    }
    writeln!(f)?;
    writeln!(f, "Per-prompt breakdown:")?;
    for (i, r) in stats.results.iter().enumerate() {
        writeln!(
            f,
            "  {:>2}. [{:<15}] {:>6} µs  {}",
            i + 1,
            r.layer.as_str(),
            r.latency.as_micros(),
            truncate(&r.prompt, 60)
        )?;
    }
    writeln!(f)?;
    writeln!(f, "Generated by: isartor v{}", env!("CARGO_PKG_VERSION"))?;

    tracing::info!(path = %path, deflection_pct = stats.deflection_pct, "Demo result written");
    Ok(())
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Parse the bundled JSONL fixture into a vec of prompts.
fn parse_replay_fixture() -> anyhow::Result<Vec<ReplayPrompt>> {
    let mut prompts = Vec::new();
    for line in REPLAY_JSONL.as_bytes().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rp: ReplayPrompt = serde_json::from_str(trimmed)?;
        prompts.push(rp);
    }
    Ok(prompts)
}

/// Seed caches with canonical prompt/response pairs.
///
/// - Inserts into L1a exact cache.
/// - Generates embeddings and inserts into L1b vector cache.
async fn seed_caches(state: &Arc<AppState>) -> anyhow::Result<()> {
    let seeds = canonical_seeds();

    for (prompt, response) in &seeds {
        // L1a: exact cache
        state
            .exact_cache
            .put(prompt.to_string(), response.to_string());

        // L1b: vector cache — generate embedding and store.
        let embedding = state
            .text_embedder
            .generate_embedding(prompt)
            .map_err(|e| anyhow::anyhow!("Embedding failed for seed prompt: {e}"))?;
        state
            .vector_cache
            .insert(embedding, response.to_string(), None)
            .await;
    }

    tracing::info!(
        seed_count = seeds.len(),
        "Demo: seeded L1a + L1b caches with canonical prompts"
    );
    Ok(())
}

/// Check a prompt against L1a (exact) then L1b (semantic) caches.
async fn check_cache_layers(state: &Arc<AppState>, prompt: &str) -> DemoHitLayer {
    // L1a: exact match
    if state.exact_cache.get(prompt).is_some() {
        return DemoHitLayer::ExactCache;
    }

    // L1b: semantic match
    let embedding = match state.text_embedder.generate_embedding(prompt) {
        Ok(e) => e,
        Err(_) => return DemoHitLayer::Passthrough,
    };

    if state.vector_cache.search(&embedding, None).await.is_some() {
        return DemoHitLayer::SemanticCache;
    }

    DemoHitLayer::Passthrough
}

/// Truncate a string for display, appending "…" if trimmed.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

async fn run_live_showcase(state: &Arc<AppState>) -> LiveShowcaseResult {
    if let Some(reason) = live_showcase_skip_reason(&state.config) {
        return LiveShowcaseResult::Skipped { reason };
    }

    let provider = state.llm_agent.provider_name().to_string();
    let model = configured_demo_model(&state.config);
    let prompt = "You are part of the Isartor post-install demo. In one short sentence, explain how a prompt firewall helps AI coding tools.";
    let start = Instant::now();

    match state.llm_agent.chat(prompt).await {
        Ok(response) => LiveShowcaseResult::Success {
            provider,
            model,
            latency: start.elapsed(),
            response_preview: truncate(response.trim(), 140),
        },
        Err(err) => LiveShowcaseResult::Failed {
            provider,
            model,
            error: truncate(&format!("{err:#}"), 140),
        },
    }
}

fn live_showcase_skip_reason(config: &AppConfig) -> Option<String> {
    if config.offline_mode {
        return Some("offline mode is active".to_string());
    }

    if provider_requires_key(&config.llm_provider) && config.external_llm_api_key.trim().is_empty()
    {
        return Some(format!(
            "no {} API key configured — run `isartor set-key -p {}` to unlock the live showcase",
            config.llm_provider, config.llm_provider
        ));
    }

    None
}

fn provider_requires_key(provider: &LlmProvider) -> bool {
    !matches!(provider, LlmProvider::Ollama)
}

fn configured_demo_model(config: &AppConfig) -> String {
    match config.llm_provider {
        LlmProvider::Azure if !config.azure_deployment_id.trim().is_empty() => format!(
            "{} (deployment; model {})",
            config.azure_deployment_id, config.external_llm_model
        ),
        _ => config.external_llm_model.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fixture_returns_50_prompts() {
        let prompts = parse_replay_fixture().unwrap();
        assert_eq!(prompts.len(), 50);
    }

    #[test]
    fn canonical_seeds_are_non_empty() {
        let seeds = canonical_seeds();
        assert!(!seeds.is_empty());
        for (p, r) in &seeds {
            assert!(!p.is_empty());
            assert!(!r.is_empty());
        }
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let long = "a".repeat(100);
        let result = truncate(&long, 20);
        assert!(result.chars().count() <= 20);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn live_showcase_skips_without_key_for_remote_provider() {
        let mut config = AppConfig::load_with_validation(false).unwrap();
        config.llm_provider = LlmProvider::Groq;
        config.external_llm_api_key.clear();

        let reason = live_showcase_skip_reason(&config).unwrap();
        assert!(reason.contains("isartor set-key -p groq"));
    }

    #[test]
    fn configured_demo_model_prefers_azure_deployment_name() {
        let mut config = AppConfig::load_with_validation(false).unwrap();
        config.llm_provider = LlmProvider::Azure;
        config.azure_deployment_id = "demo-deployment".into();
        config.external_llm_model = "gpt-4o-mini".into();

        assert_eq!(
            configured_demo_model(&config),
            "demo-deployment (deployment; model gpt-4o-mini)"
        );
    }
}

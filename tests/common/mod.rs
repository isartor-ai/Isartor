// =============================================================================
// tests/common — Shared test fixtures, mock agents, and state builders.
//
// Usage from any test file:
//   mod common;
//   use common::*;
// =============================================================================

pub mod gateway;

use std::sync::Arc;

use isartor::config::{AppConfig, CacheMode};
use isartor::state::{AppLlmAgent, AppState};

// ── Mock Agents ──────────────────────────────────────────────────────

/// Mock agent that echoes the prompt back.
pub struct EchoAgent;

#[async_trait::async_trait]
impl AppLlmAgent for EchoAgent {
    async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
        Ok(format!("echo: {prompt}"))
    }
    fn provider_name(&self) -> &'static str {
        "mock-echo"
    }
}

/// Mock agent that always succeeds with a fixed response.
pub struct SuccessAgent(pub &'static str);

#[async_trait::async_trait]
impl AppLlmAgent for SuccessAgent {
    async fn chat(&self, _prompt: &str) -> anyhow::Result<String> {
        Ok(self.0.to_string())
    }
    fn provider_name(&self) -> &'static str {
        "mock-success"
    }
}

/// Mock agent that always fails with the given error message.
pub struct FailAgent(pub &'static str);

#[async_trait::async_trait]
impl AppLlmAgent for FailAgent {
    async fn chat(&self, _prompt: &str) -> anyhow::Result<String> {
        Err(anyhow::anyhow!("{}", self.0))
    }
    fn provider_name(&self) -> &'static str {
        "mock-fail"
    }
}

/// Mock agent that counts calls via an atomic counter.
pub struct CountingAgent {
    pub response: String,
    pub counter: Arc<std::sync::atomic::AtomicU32>,
}

#[async_trait::async_trait]
impl AppLlmAgent for CountingAgent {
    async fn chat(&self, _prompt: &str) -> anyhow::Result<String> {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(self.response.clone())
    }
    fn provider_name(&self) -> &'static str {
        "mock-counting"
    }
}

// ── Config Builders ──────────────────────────────────────────────────

/// Build a test `AppConfig` with the given cache mode and sidecar URL.
pub fn test_config(mode: CacheMode, sidecar_url: &str) -> Arc<AppConfig> {
    let mut cfg = AppConfig::test_default();
    cfg.cache_mode = mode;
    cfg.layer2.sidecar_url = sidecar_url.into();
    Arc::new(cfg)
}

/// Build a test `AppConfig` with a minimal exact-only cache.
pub fn test_config_exact(sidecar_url: &str) -> Arc<AppConfig> {
    test_config(CacheMode::Exact, sidecar_url)
}

/// Build a test `AppConfig` with `enable_slm_router = true` for L2 triage tests.
pub fn test_config_slm_enabled(mode: CacheMode, sidecar_url: &str) -> Arc<AppConfig> {
    let mut cfg = (*test_config(mode, sidecar_url)).clone();
    cfg.enable_slm_router = true;
    Arc::new(cfg)
}

// ── State Builder ────────────────────────────────────────────────────

/// Build a test `AppState` with the given agent and config.
pub fn build_state(
    agent: Arc<dyn AppLlmAgent>,
    config: Arc<AppConfig>,
    embedder: Arc<isartor::layer1::embeddings::TextEmbedder>,
) -> Arc<AppState> {
    use isartor::clients::slm::SlmClient;
    use isartor::core::context_compress::InstructionCache;
    use isartor::core::usage::UsageTracker;
    use isartor::layer1::layer1a_cache::ExactMatchCache;
    use isartor::vector_cache::VectorCache;
    use std::num::NonZeroUsize;

    Arc::new(AppState {
        http_client: reqwest::Client::new(),
        exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
        vector_cache: Arc::new(VectorCache::new(
            config.similarity_threshold,
            config.cache_ttl_secs,
            config.cache_max_capacity,
        )),
        provider_chain: Arc::new(isartor::state::resolved_provider_chain(&config)),
        usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
        llm_agent: agent,
        slm_client: Arc::new(SlmClient::new(&config.layer2)),
        text_embedder: embedder,
        instruction_cache: Arc::new(InstructionCache::new()),
        provider_health: Arc::new(isartor::state::ProviderHealthTracker::from_config(&config)),
        provider_key_pools: Arc::new(isartor::state::ProviderKeyPoolManager::from_provider_chain(
            isartor::state::resolved_provider_chain(&config).as_slice(),
        )),
        config,
        #[cfg(feature = "embedded-inference")]
        embedded_classifier: None,
    })
}

/// Build a test `AppState` with the echo agent and exact caching.
pub fn echo_state(sidecar_url: &str) -> Arc<AppState> {
    let config = test_config_exact(sidecar_url);
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    build_state(Arc::new(EchoAgent), config, embedder)
}

// ── JSON helpers ─────────────────────────────────────────────────────

/// Build a JSON body `{ "prompt": "..." }` for test requests.
pub fn json_body(prompt: &str) -> axum::body::Body {
    axum::body::Body::from(serde_json::to_vec(&serde_json::json!({ "prompt": prompt })).unwrap())
}

/// Build the OpenAI chat-completion JSON fixture for wiremock.
pub fn chat_completion_json(content: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": { "content": content }
        }]
    })
}

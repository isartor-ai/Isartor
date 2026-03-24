// =============================================================================
// tests/common — Shared test fixtures, mock agents, and state builders.
//
// Usage from any test file:
//   mod common;
//   use common::*;
// =============================================================================

pub mod gateway;

use std::num::NonZeroUsize;
use std::sync::Arc;

use isartor::clients::slm::SlmClient;
use isartor::config::{
    AppConfig, CacheBackend, CacheMode, ClassifierMode, EmbeddingSidecarSettings,
    InferenceEngineMode, Layer2Settings, RouterBackend,
};
use isartor::layer1::layer1a_cache::ExactMatchCache;
use isartor::state::{AppLlmAgent, AppState};
use isartor::vector_cache::VectorCache;

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
    Arc::new(AppConfig {
        host_port: "127.0.0.1:0".into(),
        inference_engine: InferenceEngineMode::Sidecar,
        gateway_api_key: "test-key".into(),
        cache_mode: mode,
        cache_backend: CacheBackend::Memory,
        redis_url: "redis://127.0.0.1:6379".into(),
        router_backend: RouterBackend::Embedded,
        vllm_url: "http://127.0.0.1:8000".into(),
        vllm_model: "gemma-2-2b-it".into(),
        embedding_model: "all-minilm".into(),
        similarity_threshold: 0.85,
        cache_ttl_secs: 300,
        cache_max_capacity: 100,
        layer2: Layer2Settings {
            sidecar_url: sidecar_url.into(),
            model_name: "phi-3-mini".into(),
            timeout_seconds: 5,
            classifier_mode: ClassifierMode::Tiered,
            max_answer_tokens: 2048,
        },
        local_slm_url: "http://localhost:11434/api/generate".into(),
        local_slm_model: "llama3".into(),
        embedding_sidecar: EmbeddingSidecarSettings {
            sidecar_url: "http://127.0.0.1:8082".into(),
            model_name: "test".into(),
            timeout_seconds: 5,
        },
        llm_provider: "openai".into(),
        external_llm_url: "http://localhost".into(),
        external_llm_model: "gpt-4o-mini".into(),
        external_llm_api_key: "".into(),
        l3_timeout_secs: 120,
        azure_deployment_id: "".into(),
        azure_api_version: "".into(),
        enable_monitoring: false,
        enable_slm_router: false,
        otel_exporter_endpoint: "http://localhost:4317".into(),
        offline_mode: false,
        proxy_port: "0.0.0.0:8081".into(),
    })
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
    Arc::new(AppState {
        http_client: reqwest::Client::new(),
        exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
        vector_cache: Arc::new(VectorCache::new(
            config.similarity_threshold,
            config.cache_ttl_secs,
            config.cache_max_capacity,
        )),
        llm_agent: agent,
        slm_client: Arc::new(SlmClient::new(&config.layer2)),
        text_embedder: embedder,
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

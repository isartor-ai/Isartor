#![allow(dead_code)]
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::{anthropic, azure, openai, xai};
use tokio::sync::RwLock;

use crate::clients::slm::SlmClient;
use crate::config::AppConfig;
use crate::layer1::embeddings::TextEmbedder;
use crate::vector_cache::VectorCache;

// ── Exact-match cache ────────────────────────────────────────────────

struct ExactEntry {
    response: String,
    created_at: Instant,
}

pub struct ExactCache {
    entries: RwLock<HashMap<String, ExactEntry>>,
    ttl: Duration,
    max_capacity: usize,
}

impl ExactCache {
    pub fn new(ttl_secs: u64, max_capacity: u64) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_secs),
            max_capacity: max_capacity as usize,
        }
    }

    pub async fn get(&self, key: &str) -> Option<String> {
        let entries = self.entries.read().await;
        if let Some(entry) = entries.get(key) {
            if entry.created_at.elapsed() <= self.ttl {
                return Some(entry.response.clone());
            }
        }
        None
    }

    pub async fn insert(&self, key: String, response: String) {
        let mut entries = self.entries.write().await;
        let now = Instant::now();
        entries.retain(|_, e| e.created_at.elapsed() <= self.ttl);
        if entries.len() >= self.max_capacity {
            if let Some(oldest_key) = entries
                .iter()
                .min_by_key(|(_, e)| e.created_at)
                .map(|(k, _)| k.clone())
            {
                entries.remove(&oldest_key);
            }
        }
        entries.insert(
            key,
            ExactEntry {
                response,
                created_at: now,
            },
        );
    }
}

// ── Multi-provider Agent Wrapper ─────────────────────────────────────

#[async_trait::async_trait]
pub trait AppLlmAgent: Send + Sync {
    async fn chat(&self, prompt: &str) -> anyhow::Result<String>;
    fn provider_name(&self) -> &'static str;
}

pub struct RigAgent<M: rig::completion::CompletionModel> {
    pub name: &'static str,
    pub agent: Agent<M>,
}

#[async_trait::async_trait]
impl<M> AppLlmAgent for RigAgent<M>
where
    M: rig::completion::CompletionModel + Send + Sync,
{
    async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
        self.agent
            .prompt(prompt)
            .await
            .map_err(|e| anyhow::anyhow!(e))
    }

    fn provider_name(&self) -> &'static str {
        self.name
    }
}

// ── App State ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub http_client: reqwest::Client,
    pub exact_cache: Arc<ExactCache>,
    pub vector_cache: Arc<VectorCache>,

    /// Rig AI Agent encapsulating the configured Layer 3 provider.
    pub llm_agent: Arc<dyn AppLlmAgent>,

    /// Dedicated HTTP client for the llama.cpp generation sidecar.
    pub slm_client: Arc<SlmClient>,

    /// In-process sentence embedding model for Layer 1 semantic cache.
    /// Uses fastembed (ONNX Runtime) with BAAI/bge-small-en-v1.5.
    pub text_embedder: Arc<TextEmbedder>,

    #[cfg(feature = "embedded-inference")]
    pub embedded_classifier: Option<Arc<crate::services::local_inference::EmbeddedClassifier>>,
}

impl AppState {
    pub fn new(config: Arc<AppConfig>, text_embedder: Arc<TextEmbedder>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build reqwest client");

        let exact_cache = Arc::new(ExactCache::new(
            config.cache_ttl_secs,
            config.cache_max_capacity,
        ));
        let vector_cache = Arc::new(VectorCache::new(
            config.similarity_threshold,
            config.cache_ttl_secs,
            config.cache_max_capacity,
        ));

        let agent: Arc<dyn AppLlmAgent> = match config.llm_provider.as_str() {
            "azure" => {
                let client: azure::Client = azure::Client::builder()
                    .api_key(config.external_llm_api_key.as_str())
                    .azure_endpoint(config.external_llm_url.clone())
                    .api_version(&config.azure_api_version)
                    .build()
                    .expect("Failed to initialize Azure OpenAI client");
                Arc::new(RigAgent {
                    name: "azure",
                    agent: client.agent(&config.azure_deployment_id).build(),
                })
            }
            "anthropic" => {
                let client = anthropic::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Anthropic client");
                Arc::new(RigAgent {
                    name: "anthropic",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "xai" => {
                let client = xai::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize xAI client");
                Arc::new(RigAgent {
                    name: "xai",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            _ => {
                let client = openai::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize OpenAI client");
                Arc::new(RigAgent {
                    name: "openai",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
        };

        let slm_client = Arc::new(SlmClient::new(&config.layer2));

        #[cfg(feature = "embedded-inference")]
        let embedded_classifier =
            if config.inference_engine == crate::config::InferenceEngineMode::Embedded {
                // NOTE: In a real app we would want to bubble up this error instead of
                // doing blocking initialization or panic, but for the sake of the architecture
                // state encapsulation we can block_on it or pass it in. Assuming blocking for now
                // or we change AppState::new to be async.
                let mut cfg = crate::services::local_inference::EmbeddedClassifierConfig::default();
                // Allow overriding the model path via env var (e.g. Docker image with baked-in model).
                if let Ok(path) = std::env::var("ISARTOR__EMBEDDED__MODEL_PATH") {
                    if !path.is_empty() {
                        cfg.model_path = Some(path);
                    }
                }
                // Since `AppState::new` is not async, we use a blocking fallback or expect initialization elsewhere.
                // For simplicity in this sync constructor we will leave it as None and assume an async `init` method later,
                // or just block_on. We use block_on here for convenience.
                let engine = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        crate::services::local_inference::EmbeddedClassifier::new(cfg).await
                    })
                })
                .expect("Failed to initialize Embedded Classifier");
                Some(Arc::new(engine))
            } else {
                None
            };

        Self {
            config,
            http_client,
            exact_cache,
            vector_cache,
            llm_agent: agent,
            slm_client,
            text_embedder,
            #[cfg(feature = "embedded-inference")]
            embedded_classifier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer1::embeddings::shared_test_embedder;
    // ── ExactCache tests ─────────────────────────────────────────

    #[tokio::test]
    async fn exact_cache_insert_and_get() {
        let cache = ExactCache::new(300, 100);
        cache.insert("key1".into(), "response1".into()).await;
        assert_eq!(cache.get("key1").await, Some("response1".into()));
    }

    #[tokio::test]
    async fn exact_cache_miss_returns_none() {
        let cache = ExactCache::new(300, 100);
        assert_eq!(cache.get("nonexistent").await, None);
    }

    #[tokio::test]
    async fn exact_cache_ttl_expiry() {
        // TTL of 0 seconds — entries expire immediately.
        let cache = ExactCache::new(0, 100);
        cache.insert("key1".into(), "response1".into()).await;
        // Wait a tiny bit to ensure the TTL has passed.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert_eq!(cache.get("key1").await, None);
    }

    #[tokio::test]
    async fn exact_cache_capacity_eviction() {
        let cache = ExactCache::new(300, 2);
        cache.insert("key1".into(), "r1".into()).await;
        cache.insert("key2".into(), "r2".into()).await;
        // This insert should evict the oldest (key1).
        cache.insert("key3".into(), "r3".into()).await;

        assert_eq!(cache.get("key1").await, None);
        assert_eq!(cache.get("key2").await, Some("r2".into()));
        assert_eq!(cache.get("key3").await, Some("r3".into()));
    }

    #[tokio::test]
    async fn exact_cache_overwrite_same_key() {
        let cache = ExactCache::new(300, 100);
        cache.insert("key1".into(), "old".into()).await;
        cache.insert("key1".into(), "new".into()).await;
        assert_eq!(cache.get("key1").await, Some("new".into()));
    }

    #[tokio::test]
    async fn exact_cache_expired_entries_evicted_on_insert() {
        let cache = ExactCache::new(0, 10);
        cache.insert("key1".into(), "r1".into()).await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        // Insert again — the expired entry should be evicted first.
        cache.insert("key2".into(), "r2".into()).await;
        let entries = cache.entries.read().await;
        // Only key2 should remain (key1 expired and was removed).
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key("key2"));
    }

    // ── AppLlmAgent mock for testing ─────────────────────────────

    struct MockAgent;

    #[async_trait::async_trait]
    impl AppLlmAgent for MockAgent {
        async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
            Ok(format!("Mock response to: {prompt}"))
        }
        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn mock_agent_chat() {
        let agent = MockAgent;
        let result = agent.chat("hello").await.unwrap();
        assert_eq!(result, "Mock response to: hello");
        assert_eq!(agent.provider_name(), "mock");
    }

    // ── AppState::new tests ──────────────────────────────────────

    fn make_test_config(provider: &str) -> Arc<AppConfig> {
        Arc::new(AppConfig {
            host_port: "127.0.0.1:0".into(),
            inference_engine: crate::config::InferenceEngineMode::Sidecar,
            gateway_api_key: "test-key".into(),
            llm_provider: provider.into(),
            external_llm_url: "https://api.openai.com".into(),
            external_llm_api_key: "sk-test".into(),
            external_llm_model: "gpt-4o-mini".into(),
            azure_api_version: "2024-02-15-preview".into(),
            azure_deployment_id: "my-deployment".into(),
            cache_mode: crate::config::CacheMode::Both,
            cache_ttl_secs: 300,
            cache_max_capacity: 1000,
            embedding_model: "test".into(),
            similarity_threshold: 0.92,
            enable_monitoring: false,
            otel_exporter_endpoint: String::new(),
            local_slm_url: "http://localhost:11434/api/generate".into(),
            local_slm_model: "llama3".into(),
            layer2: crate::config::Layer2Settings {
                sidecar_url: "http://localhost:8081".into(),
                model_name: "test-model".into(),
                timeout_seconds: 30,
            },
            embedding_sidecar: crate::config::EmbeddingSidecarSettings {
                sidecar_url: "http://localhost:8082".into(),
                model_name: "test-embed".into(),
                timeout_seconds: 30,
            },
            pipeline_min_concurrency: 4,
            pipeline_max_concurrency: 256,
            pipeline_target_latency_ms: 500,
            pipeline_similarity_threshold: 0.85,
            pipeline_rerank_top_k: 5,
            pipeline_embedding_dim: 128,
        })
    }

    #[tokio::test]
    async fn app_state_new_default_openai_provider() {
        let state = AppState::new(make_test_config("openai"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "openai");
        assert_eq!(state.config.llm_provider, "openai");
    }

    #[tokio::test]
    async fn app_state_new_unknown_provider_defaults_to_openai() {
        // "unknown-provider" falls to the default branch → openai.
        let state = AppState::new(make_test_config("unknown-provider"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "openai");
    }

    #[tokio::test]
    async fn app_state_new_anthropic_provider() {
        let state = AppState::new(make_test_config("anthropic"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "anthropic");
    }

    #[tokio::test]
    async fn app_state_new_xai_provider() {
        let state = AppState::new(make_test_config("xai"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "xai");
    }

    #[tokio::test]
    async fn app_state_new_azure_provider() {
        let state = AppState::new(make_test_config("azure"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "azure");
    }

    #[tokio::test]
    async fn app_state_caches_are_initialised() {
        let state = AppState::new(make_test_config("openai"), shared_test_embedder());
        // Verify caches and clients are properly initialised.
        assert!(Arc::strong_count(&state.exact_cache) >= 1);
        assert!(Arc::strong_count(&state.vector_cache) >= 1);
        assert!(Arc::strong_count(&state.slm_client) >= 1);
    }
}

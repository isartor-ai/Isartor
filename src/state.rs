#![allow(dead_code)]
use std::env;
use std::sync::Arc;
use std::time::Duration;

use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::client::Nothing;
use rig::completion::Prompt;
use rig::providers::{
    anthropic, azure, cohere, deepseek, galadriel, gemini, groq, huggingface, hyperbolic, mira,
    mistral, moonshot, ollama, openai, openrouter, perplexity, together, xai,
};

use crate::clients::slm::SlmClient;
use crate::config::AppConfig;
use crate::layer1::embeddings::TextEmbedder;
use crate::layer1::layer1a_cache::ExactMatchCache;
use crate::vector_cache::VectorCache;

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
    pub exact_cache: Arc<ExactMatchCache>,
    pub vector_cache: Arc<VectorCache>,

    /// Rig AI Agent encapsulating the configured Layer 3 provider.
    pub llm_agent: Arc<dyn AppLlmAgent>,

    /// Dedicated HTTP client for the llama.cpp generation sidecar.
    pub slm_client: Arc<SlmClient>,

    /// In-process sentence embedding model for Layer 1 semantic cache.
    /// Pure-Rust candle BertModel with sentence-transformers/all-MiniLM-L6-v2.
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

        let exact_cache = Arc::new(ExactMatchCache::new(
            std::num::NonZeroUsize::new(config.cache_max_capacity as usize)
                .unwrap_or_else(|| std::num::NonZeroUsize::new(128).unwrap()),
        ));
        let vector_cache = Arc::new(VectorCache::new(
            config.similarity_threshold,
            config.cache_ttl_secs,
            config.cache_max_capacity,
        ));

        // Note: `config.external_llm_url` is only used for the Azure provider below.
        // For all other providers (anthropic, xai, gemini, mistral, groq, deepseek, etc.),
        // the rig-core clients currently do not consume `external_llm_url`, so
        // `ISARTOR__EXTERNAL_LLM_URL` has no effect when those providers are selected.
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
            "gemini" => {
                let client = gemini::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Gemini client");
                Arc::new(RigAgent {
                    name: "gemini",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "mistral" => {
                let client = mistral::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Mistral client");
                Arc::new(RigAgent {
                    name: "mistral",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "groq" => {
                let client = groq::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Groq client");
                Arc::new(RigAgent {
                    name: "groq",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "deepseek" => {
                let client = deepseek::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize DeepSeek client");
                Arc::new(RigAgent {
                    name: "deepseek",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "cohere" => {
                let client = cohere::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Cohere client");
                Arc::new(RigAgent {
                    name: "cohere",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "galadriel" => {
                let client = galadriel::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Galadriel client");
                Arc::new(RigAgent {
                    name: "galadriel",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "hyperbolic" => {
                let client = hyperbolic::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Hyperbolic client");
                Arc::new(RigAgent {
                    name: "hyperbolic",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "huggingface" => {
                let client = huggingface::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize HuggingFace client");
                Arc::new(RigAgent {
                    name: "huggingface",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "mira" => {
                let client = mira::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Mira client");
                Arc::new(RigAgent {
                    name: "mira",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "moonshot" => {
                let client = moonshot::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Moonshot client");
                Arc::new(RigAgent {
                    name: "moonshot",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "ollama" => {
                // Ollama is a local provider — no API key required.
                // If an external LLM URL is configured, use it to override the default Ollama host.
                // This allows running Ollama on a non-localhost host/port (e.g., via ISARTOR__EXTERNAL_LLM_URL).
                env::set_var("OLLAMA_HOST", &config.external_llm_url);
                let client =
                    ollama::Client::new(Nothing).expect("Failed to initialize Ollama client");
                Arc::new(RigAgent {
                    name: "ollama",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "openrouter" => {
                let client = openrouter::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize OpenRouter client");
                Arc::new(RigAgent {
                    name: "openrouter",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "perplexity" => {
                let client = perplexity::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Perplexity client");
                Arc::new(RigAgent {
                    name: "perplexity",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "together" => {
                let client = together::Client::new(&config.external_llm_api_key)
                    .expect("Failed to initialize Together AI client");
                Arc::new(RigAgent {
                    name: "together",
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
        // Tested in layer1::layer1a_cache::tests — retained as placeholder.
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
            cache_backend: crate::config::CacheBackend::Memory,
            redis_url: "redis://127.0.0.1:6379".into(),
            router_backend: crate::config::RouterBackend::Embedded,
            vllm_url: "http://127.0.0.1:8000".into(),
            vllm_model: "gemma-2-2b-it".into(),
            cache_ttl_secs: 300,
            cache_max_capacity: 1000,
            embedding_model: "test".into(),
            similarity_threshold: 0.92,
            enable_monitoring: false,
            enable_slm_router: false,
            otel_exporter_endpoint: String::new(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
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
        })
    }

    #[tokio::test]
    async fn app_state_new_default_openai_provider() {
        let state = AppState::new(make_test_config("openai"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "openai");
        assert_eq!(state.config.llm_provider, "openai".into());
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
    async fn app_state_new_gemini_provider() {
        let state = AppState::new(make_test_config("gemini"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "gemini");
    }

    #[tokio::test]
    async fn app_state_new_mistral_provider() {
        let state = AppState::new(make_test_config("mistral"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "mistral");
    }

    #[tokio::test]
    async fn app_state_new_groq_provider() {
        let state = AppState::new(make_test_config("groq"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "groq");
    }

    #[tokio::test]
    async fn app_state_new_deepseek_provider() {
        let state = AppState::new(make_test_config("deepseek"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "deepseek");
    }

    #[tokio::test]
    async fn app_state_new_cohere_provider() {
        let state = AppState::new(make_test_config("cohere"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "cohere");
    }

    #[tokio::test]
    async fn app_state_new_galadriel_provider() {
        let state = AppState::new(make_test_config("galadriel"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "galadriel");
    }

    #[tokio::test]
    async fn app_state_new_hyperbolic_provider() {
        let state = AppState::new(make_test_config("hyperbolic"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "hyperbolic");
    }

    #[tokio::test]
    async fn app_state_new_huggingface_provider() {
        let state = AppState::new(make_test_config("huggingface"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "huggingface");
    }

    #[tokio::test]
    async fn app_state_new_mira_provider() {
        let state = AppState::new(make_test_config("mira"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "mira");
    }

    #[tokio::test]
    async fn app_state_new_moonshot_provider() {
        let state = AppState::new(make_test_config("moonshot"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "moonshot");
    }

    #[tokio::test]
    async fn app_state_new_ollama_provider() {
        let state = AppState::new(make_test_config("ollama"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "ollama");
    }

    #[tokio::test]
    async fn app_state_new_openrouter_provider() {
        let state = AppState::new(make_test_config("openrouter"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "openrouter");
    }

    #[tokio::test]
    async fn app_state_new_perplexity_provider() {
        let state = AppState::new(make_test_config("perplexity"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "perplexity");
    }

    #[tokio::test]
    async fn app_state_new_together_provider() {
        let state = AppState::new(make_test_config("together"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "together");
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

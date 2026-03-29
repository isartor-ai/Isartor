#![allow(dead_code)]
use std::env;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use parking_lot::Mutex;
use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::client::Nothing;
use rig::completion::Prompt;
use rig::providers::{
    anthropic, azure, cohere, deepseek, galadriel, gemini, groq, huggingface, hyperbolic, mira,
    mistral, moonshot, ollama, openai, openrouter, perplexity, together, xai,
};

use crate::clients::slm::SlmClient;
use crate::config::{
    AppConfig, DEFAULT_OPENAI_CHAT_COMPLETIONS_URL, LlmProvider, default_chat_completions_url,
};
use crate::core::context_compress::InstructionCache;
use crate::layer1::embeddings::TextEmbedder;
use crate::layer1::layer1a_cache::ExactMatchCache;
use crate::models::{ProviderHealthStatus, ProviderStatusEntry, ProviderStatusResponse};
use crate::providers::copilot::CopilotAgent;
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

#[derive(Debug, Clone, Copy, Default)]
enum LastProviderOutcome {
    Healthy,
    Failing,
    #[default]
    Unknown,
}

#[derive(Debug, Default)]
struct ProviderHealthState {
    requests_total: u64,
    errors_total: u64,
    last_success: Option<String>,
    last_error: Option<String>,
    last_error_message: Option<String>,
    last_outcome: LastProviderOutcome,
}

#[derive(Debug)]
pub struct ProviderHealthTracker {
    provider_name: String,
    configured_model: String,
    endpoint: String,
    api_key_configured: bool,
    endpoint_configured: bool,
    state: Mutex<ProviderHealthState>,
}

impl ProviderHealthTracker {
    pub fn from_config(config: &AppConfig) -> Self {
        let endpoint = provider_status_endpoint(config);
        let requires_api_key = !matches!(config.llm_provider, LlmProvider::Ollama);

        Self {
            provider_name: config.llm_provider.as_str().to_string(),
            configured_model: config.configured_model_id(),
            endpoint_configured: !endpoint.trim().is_empty(),
            api_key_configured: !requires_api_key || !config.external_llm_api_key.trim().is_empty(),
            endpoint,
            state: Mutex::new(ProviderHealthState::default()),
        }
    }

    pub fn record_success(&self) {
        let mut state = self.state.lock();
        state.requests_total += 1;
        state.last_success = Some(Utc::now().to_rfc3339());
        state.last_outcome = LastProviderOutcome::Healthy;
    }

    pub fn record_failure(&self, error: &str) {
        let mut state = self.state.lock();
        state.requests_total += 1;
        state.errors_total += 1;
        state.last_error = Some(Utc::now().to_rfc3339());
        state.last_error_message = Some(compact_provider_error(error));
        state.last_outcome = LastProviderOutcome::Failing;
    }

    pub fn snapshot(&self) -> ProviderStatusResponse {
        ProviderStatusResponse {
            active_provider: self.provider_name.clone(),
            providers: vec![self.snapshot_entry()],
        }
    }

    fn snapshot_entry(&self) -> ProviderStatusEntry {
        let state = self.state.lock();
        ProviderStatusEntry {
            name: self.provider_name.clone(),
            active: true,
            status: match state.last_outcome {
                LastProviderOutcome::Healthy => ProviderHealthStatus::Healthy,
                LastProviderOutcome::Failing => ProviderHealthStatus::Failing,
                LastProviderOutcome::Unknown => ProviderHealthStatus::Unknown,
            },
            model: self.configured_model.clone(),
            endpoint: self.endpoint.clone(),
            api_key_configured: self.api_key_configured,
            endpoint_configured: self.endpoint_configured,
            requests_total: state.requests_total,
            errors_total: state.errors_total,
            last_success: state.last_success.clone(),
            last_error: state.last_error.clone(),
            last_error_message: state.last_error_message.clone(),
        }
    }
}

macro_rules! build_rig_agent {
    ($name:literal, $client:path, $api_key:expr, $model:expr, $http_client:expr) => {{
        let client = <$client>::builder()
            .api_key($api_key.clone())
            .http_client($http_client.clone())
            .build()
            .expect(concat!("Failed to initialize ", $name, " client"));
        Arc::new(RigAgent {
            name: $name,
            agent: client.agent($model).build(),
        })
    }};
}

fn is_openai_compatible_runtime_provider(provider: &str) -> bool {
    matches!(
        provider,
        "openai" | "cerebras" | "nebius" | "siliconflow" | "fireworks" | "nvidia" | "chutes"
    )
}

fn openai_compatible_base_url(endpoint: &str) -> String {
    endpoint
        .trim()
        .trim_end_matches('/')
        .trim_end_matches("/chat/completions")
        .to_string()
}

fn provider_status_endpoint(config: &AppConfig) -> String {
    match &config.llm_provider {
        LlmProvider::Azure => {
            if config.external_llm_url.trim().is_empty()
                || config.azure_deployment_id.trim().is_empty()
                || config.azure_api_version.trim().is_empty()
            {
                config.external_llm_url.trim().to_string()
            } else {
                format!(
                    "{}/openai/deployments/{}/chat/completions?api-version={}",
                    config.external_llm_url.trim_end_matches('/'),
                    config.azure_deployment_id,
                    config.azure_api_version
                )
            }
        }
        LlmProvider::Anthropic => "https://api.anthropic.com/v1/messages".to_string(),
        LlmProvider::Copilot => {
            let configured = config.external_llm_url.trim();
            if configured.is_empty() {
                "https://api.githubcopilot.com/chat/completions".to_string()
            } else {
                configured.to_string()
            }
        }
        LlmProvider::Gemini => format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}",
            config.external_llm_model
        ),
        LlmProvider::Ollama => {
            let configured = config.external_llm_url.trim();
            if configured.is_empty() {
                "http://localhost:11434".to_string()
            } else {
                configured.to_string()
            }
        }
        LlmProvider::Cohere => "https://api.cohere.ai/v1/chat".to_string(),
        LlmProvider::Huggingface => format!(
            "https://api-inference.huggingface.co/models/{}",
            config.external_llm_model
        ),
        provider => {
            let configured = config.external_llm_url.trim();
            if let Some(default_url) = default_chat_completions_url(provider) {
                if configured.is_empty()
                    || (*provider != LlmProvider::Openai
                        && configured == DEFAULT_OPENAI_CHAT_COMPLETIONS_URL)
                {
                    default_url.to_string()
                } else {
                    configured.to_string()
                }
            } else {
                configured.to_string()
            }
        }
    }
}

fn compact_provider_error(error: &str) -> String {
    let single_line = error.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX_LEN: usize = 240;
    if single_line.len() <= MAX_LEN {
        single_line
    } else {
        format!("{}...", &single_line[..MAX_LEN - 3])
    }
}

fn build_openai_compatible_rig_agent(
    name: &'static str,
    api_key: &str,
    model: &str,
    endpoint: &str,
    http_client: rig::http_client::ReqwestClient,
) -> Arc<dyn AppLlmAgent> {
    let client = openai::Client::builder()
        .api_key(api_key)
        .base_url(openai_compatible_base_url(endpoint))
        .http_client(http_client)
        .build()
        .expect("Failed to initialize OpenAI-compatible client");
    Arc::new(RigAgent {
        name,
        agent: client.agent(model).build(),
    })
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
    pub provider_health: Arc<ProviderHealthTracker>,

    /// Rig AI Agent encapsulating the configured Layer 3 provider.
    pub llm_agent: Arc<dyn AppLlmAgent>,

    /// Dedicated HTTP client for the llama.cpp generation sidecar.
    pub slm_client: Arc<SlmClient>,

    /// In-process sentence embedding model for Layer 1 semantic cache.
    /// Pure-Rust candle BertModel with sentence-transformers/all-MiniLM-L6-v2.
    pub text_embedder: Arc<TextEmbedder>,

    /// L2.5 instruction dedup cache for cross-turn session deduplication.
    pub instruction_cache: Arc<InstructionCache>,

    #[cfg(feature = "embedded-inference")]
    pub embedded_classifier: Option<Arc<crate::services::local_inference::EmbeddedClassifier>>,
}

impl AppState {
    pub fn new(config: Arc<AppConfig>, text_embedder: Arc<TextEmbedder>) -> Self {
        let l3_timeout = Duration::from_secs(config.l3_timeout_secs);
        let http_client = reqwest::Client::builder()
            .timeout(l3_timeout)
            .build()
            .expect("failed to build reqwest client");
        let rig_http_client = rig::http_client::ReqwestClient::builder()
            .timeout(l3_timeout)
            .build()
            .expect("failed to build rig reqwest client");

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
                    .http_client(rig_http_client.clone())
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
                build_rig_agent!(
                    "anthropic",
                    anthropic::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "copilot" => Arc::new(CopilotAgent::new(
                http_client.clone(),
                config.external_llm_api_key.clone(),
                config.external_llm_model.clone(),
                l3_timeout,
            )),
            "xai" => {
                build_rig_agent!(
                    "xai",
                    xai::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "gemini" => {
                build_rig_agent!(
                    "gemini",
                    gemini::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "mistral" => {
                build_rig_agent!(
                    "mistral",
                    mistral::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "groq" => {
                build_rig_agent!(
                    "groq",
                    groq::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "deepseek" => {
                build_rig_agent!(
                    "deepseek",
                    deepseek::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "cohere" => {
                build_rig_agent!(
                    "cohere",
                    cohere::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "galadriel" => {
                build_rig_agent!(
                    "galadriel",
                    galadriel::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "hyperbolic" => {
                build_rig_agent!(
                    "hyperbolic",
                    hyperbolic::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "huggingface" => {
                build_rig_agent!(
                    "huggingface",
                    huggingface::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "mira" => {
                build_rig_agent!(
                    "mira",
                    mira::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "moonshot" => {
                build_rig_agent!(
                    "moonshot",
                    moonshot::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "ollama" => {
                // Ollama is a local provider — no API key required.
                // If an external LLM URL is configured, use it to override the default Ollama host.
                // This allows running Ollama on a non-localhost host/port (e.g., via ISARTOR__EXTERNAL_LLM_URL).
                // AppState::new() runs during startup before any worker threads are spawned,
                // so updating the process environment here is safe under Rust 2024 rules.
                unsafe {
                    env::set_var("OLLAMA_HOST", &config.external_llm_url);
                }
                let client = ollama::Client::builder()
                    .api_key(Nothing)
                    .http_client(rig_http_client.clone())
                    .build()
                    .expect("Failed to initialize Ollama client");
                Arc::new(RigAgent {
                    name: "ollama",
                    agent: client.agent(&config.external_llm_model).build(),
                })
            }
            "openrouter" => {
                build_rig_agent!(
                    "openrouter",
                    openrouter::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "perplexity" => {
                build_rig_agent!(
                    "perplexity",
                    perplexity::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            "together" => {
                build_rig_agent!(
                    "together",
                    together::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
            }
            provider if is_openai_compatible_runtime_provider(provider) => {
                build_openai_compatible_rig_agent(
                    provider,
                    &config.external_llm_api_key,
                    &config.external_llm_model,
                    &config.external_llm_url,
                    rig_http_client,
                )
            }
            _ => {
                build_rig_agent!(
                    "openai",
                    openai::Client,
                    config.external_llm_api_key,
                    &config.external_llm_model,
                    rig_http_client
                )
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
                if let Ok(path) = std::env::var("ISARTOR__EMBEDDED__MODEL_PATH")
                    && !path.is_empty()
                {
                    cfg.model_path = Some(path);
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
            config: config.clone(),
            http_client,
            exact_cache,
            vector_cache,
            provider_health: Arc::new(ProviderHealthTracker::from_config(&config)),
            llm_agent: agent,
            slm_client,
            text_embedder,
            instruction_cache: Arc::new(InstructionCache::new()),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier,
        }
    }

    pub async fn chat_with_model(&self, prompt: &str, model: &str) -> anyhow::Result<String> {
        if model == self.config.configured_model_id() {
            return self.llm_agent.chat(prompt).await;
        }

        let l3_timeout = Duration::from_secs(self.config.l3_timeout_secs);
        let rig_http_client = rig::http_client::ReqwestClient::builder()
            .timeout(l3_timeout)
            .build()
            .expect("failed to build rig reqwest client");

        match self.config.llm_provider.as_str() {
            "azure" => {
                let client: azure::Client = azure::Client::builder()
                    .api_key(self.config.external_llm_api_key.as_str())
                    .http_client(rig_http_client)
                    .azure_endpoint(self.config.external_llm_url.clone())
                    .api_version(&self.config.azure_api_version)
                    .build()
                    .expect("Failed to initialize Azure OpenAI client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "anthropic" => {
                let client = anthropic::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize anthropic client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "copilot" => {
                CopilotAgent::chat_with_model(
                    &self.http_client,
                    &self.config.external_llm_api_key,
                    model,
                    l3_timeout,
                    prompt,
                )
                .await
            }
            "xai" => {
                let client = xai::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize xai client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "gemini" => {
                let client = gemini::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize gemini client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "mistral" => {
                let client = mistral::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize mistral client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "groq" => {
                let client = groq::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize groq client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "deepseek" => {
                let client = deepseek::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize deepseek client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "cohere" => {
                let client = cohere::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize cohere client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "galadriel" => {
                let client = galadriel::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize galadriel client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "hyperbolic" => {
                let client = hyperbolic::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize hyperbolic client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "huggingface" => {
                let client = huggingface::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize huggingface client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "mira" => {
                let client = mira::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize mira client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "moonshot" => {
                let client = moonshot::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize moonshot client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "ollama" => {
                unsafe {
                    env::set_var("OLLAMA_HOST", &self.config.external_llm_url);
                }
                let client = ollama::Client::builder()
                    .api_key(Nothing)
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize Ollama client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "openrouter" => {
                let client = openrouter::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize openrouter client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "perplexity" => {
                let client = perplexity::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize perplexity client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            "together" => {
                let client = together::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize together client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            provider if is_openai_compatible_runtime_provider(provider) => {
                let client = openai::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .base_url(openai_compatible_base_url(&self.config.external_llm_url))
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize OpenAI-compatible client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
            _ => {
                let client = openai::Client::builder()
                    .api_key(self.config.external_llm_api_key.clone())
                    .http_client(rig_http_client)
                    .build()
                    .expect("Failed to initialize openai client");
                client
                    .agent(model)
                    .build()
                    .prompt(prompt)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))
            }
        }
    }

    pub fn record_provider_success(&self) {
        self.provider_health.record_success();
    }

    pub fn record_provider_failure(&self, error: &str) {
        self.provider_health.record_failure(error);
    }

    pub fn provider_status(&self) -> ProviderStatusResponse {
        self.provider_health.snapshot()
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
            model_aliases: std::collections::HashMap::new(),
            l3_timeout_secs: 120,
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
            enable_request_logs: false,
            request_log_path: "~/.isartor/request_logs".into(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
            local_slm_url: "http://localhost:11434/api/generate".into(),
            local_slm_model: "llama3".into(),
            layer2: crate::config::Layer2Settings {
                sidecar_url: "http://localhost:8081".into(),
                model_name: "test-model".into(),
                timeout_seconds: 30,
                classifier_mode: crate::config::ClassifierMode::Tiered,
                max_answer_tokens: 2048,
            },
            embedding_sidecar: crate::config::EmbeddingSidecarSettings {
                sidecar_url: "http://localhost:8082".into(),
                model_name: "test-embed".into(),
                timeout_seconds: 30,
            },
            enable_context_optimizer: true,
            context_optimizer_dedup: true,
            context_optimizer_minify: true,
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
    async fn app_state_new_cerebras_provider() {
        let mut config = (*make_test_config("cerebras")).clone();
        config.external_llm_url = "https://api.cerebras.ai/v1/chat/completions".into();
        let state = AppState::new(Arc::new(config), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "cerebras");
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
    async fn app_state_new_copilot_provider() {
        let state = AppState::new(make_test_config("copilot"), shared_test_embedder());
        assert_eq!(state.llm_agent.provider_name(), "copilot");
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

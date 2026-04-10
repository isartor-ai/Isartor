use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

/// Inference Engine mode
///
/// Set via `ISARTOR_INFERENCE_ENGINE` env var.
///
/// * `"sidecar"`  - Uses external API calls (e.g. to llama.cpp sidecar) for inference. (Default)
/// * `"embedded"` - Uses embedded Candle engine for inference in-process. Requires `embedded-inference` feature.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum InferenceEngineMode {
    #[default]
    Sidecar,
    Embedded,
}

/// Cache operating mode.
///
/// Set via `ISARTOR_CACHE_MODE` env var.
///
/// * `"exact"`    — SHA-256 hash of the prompt; only identical prompts hit.
/// * `"semantic"` — Cosine similarity on embedding vectors.
/// * `"both"`     — Exact match is checked first (fast), then semantic.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheMode {
    Exact,
    Semantic,
    #[default]
    Both,
}

/// Supported external LLM providers.
///
/// This is used for the `llm_provider` configuration field. The string values
/// are deserialized in a case-insensitive (lowercase) manner via Serde. Any
/// unsupported provider string will cause configuration loading to fail,
/// avoiding silent fallbacks to unintended providers.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    /// Default provider if none is specified explicitly.
    #[default]
    Openai,
    Azure,
    Anthropic,
    Copilot,
    Xai,
    Gemini,
    Mistral,
    Groq,
    Cerebras,
    Nebius,
    Siliconflow,
    Fireworks,
    Nvidia,
    Chutes,
    Deepseek,
    Cohere,
    Galadriel,
    Hyperbolic,
    Huggingface,
    Mira,
    Moonshot,
    Ollama,
    Openrouter,
    Perplexity,
    Together,
}

impl LlmProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            LlmProvider::Openai => "openai",
            LlmProvider::Azure => "azure",
            LlmProvider::Anthropic => "anthropic",
            LlmProvider::Copilot => "copilot",
            LlmProvider::Xai => "xai",
            LlmProvider::Gemini => "gemini",
            LlmProvider::Mistral => "mistral",
            LlmProvider::Groq => "groq",
            LlmProvider::Cerebras => "cerebras",
            LlmProvider::Nebius => "nebius",
            LlmProvider::Siliconflow => "siliconflow",
            LlmProvider::Fireworks => "fireworks",
            LlmProvider::Nvidia => "nvidia",
            LlmProvider::Chutes => "chutes",
            LlmProvider::Deepseek => "deepseek",
            LlmProvider::Cohere => "cohere",
            LlmProvider::Galadriel => "galadriel",
            LlmProvider::Hyperbolic => "hyperbolic",
            LlmProvider::Huggingface => "huggingface",
            LlmProvider::Mira => "mira",
            LlmProvider::Moonshot => "moonshot",
            LlmProvider::Ollama => "ollama",
            LlmProvider::Openrouter => "openrouter",
            LlmProvider::Perplexity => "perplexity",
            LlmProvider::Together => "together",
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum KeyRotationStrategy {
    #[default]
    RoundRobin,
    Priority,
}

fn default_key_cooldown_secs() -> u64 {
    60
}

fn default_provider_key_priority() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct ProviderPricingConfig {
    #[serde(default)]
    pub input_cost_per_million_usd: f64,
    #[serde(default)]
    pub output_cost_per_million_usd: f64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuotaLimitAction {
    #[default]
    Warn,
    Block,
    Fallback,
}

fn default_quota_warning_threshold_ratio() -> f64 {
    0.8
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct ProviderQuotaConfig {
    #[serde(default)]
    pub daily_token_limit: Option<u64>,
    #[serde(default)]
    pub weekly_token_limit: Option<u64>,
    #[serde(default)]
    pub monthly_token_limit: Option<u64>,
    #[serde(default)]
    pub daily_cost_limit_usd: Option<f64>,
    #[serde(default)]
    pub weekly_cost_limit_usd: Option<f64>,
    #[serde(default)]
    pub monthly_cost_limit_usd: Option<f64>,
    #[serde(default)]
    pub action_on_limit: QuotaLimitAction,
    #[serde(default = "default_quota_warning_threshold_ratio")]
    pub warning_threshold_ratio: f64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
pub struct ProviderKeyConfig {
    #[serde(default)]
    pub key: String,
    #[serde(default = "default_provider_key_priority")]
    pub priority: u32,
    #[serde(default)]
    pub label: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
pub struct FallbackProviderConfig {
    pub provider: LlmProvider,
    pub model: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub provider_keys: Vec<ProviderKeyConfig>,
    #[serde(default)]
    pub key_rotation_strategy: KeyRotationStrategy,
    #[serde(default = "default_key_cooldown_secs")]
    pub key_cooldown_secs: u64,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub azure_deployment_id: String,
    #[serde(default = "default_azure_api_version")]
    pub azure_api_version: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct ClassifierRoutingRuleConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub task_type: Option<String>,
    #[serde(default)]
    pub complexity: Option<String>,
    #[serde(default)]
    pub persona: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub min_confidence: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ClassifierRoutingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub artifacts_path: String,
    #[serde(default = "default_classifier_routing_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_classifier_routing_fallback_to_existing")]
    pub fallback_to_existing_routing: bool,
    #[serde(default)]
    pub rules: Vec<ClassifierRoutingRuleConfig>,
    /// Model matrix: a 2D mapping of `complexity → task_type → "provider/model"`.
    ///
    /// Provides a visual grid-style routing config that compiles into rules at
    /// startup.  Explicit `rules` take priority; matrix-derived rules are
    /// appended after, ordered most-specific first.
    ///
    /// Use the special value `"local"` to indicate that a cell should stay on
    /// the local cache/SLM path (no L3 provider override).
    ///
    /// Example (TOML):
    /// ```toml
    /// [classifier_routing.matrix.complex]
    /// code_generation = "groq/llama-3.3-70b-versatile"
    /// analysis        = "anthropic/claude-sonnet-4-20250514"
    /// default         = "openai/gpt-4o"
    ///
    /// [classifier_routing.matrix.simple]
    /// code_generation = "groq/llama-3.1-8b-instant"
    /// default         = "local"
    /// ```
    #[serde(default)]
    pub matrix: HashMap<String, HashMap<String, String>>,
}

impl ClassifierRoutingConfig {
    /// Returns the merged rule list: explicit `rules` first, then matrix-
    /// derived rules sorted most-specific → least-specific.
    ///
    /// The `"default"` key in either dimension becomes a wildcard (None).
    /// The `"local"` target value is skipped (means "no provider override").
    pub fn effective_rules(&self) -> Vec<ClassifierRoutingRuleConfig> {
        if self.matrix.is_empty() {
            return self.rules.clone();
        }

        let mut matrix_rules: Vec<(u8, ClassifierRoutingRuleConfig)> = Vec::new();

        for (complexity_key, task_map) in &self.matrix {
            for (task_key, target) in task_map {
                let target = target.trim();
                if target.is_empty() || target.eq_ignore_ascii_case("local") {
                    continue;
                }

                let (provider, model) = if let Some((p, m)) = target.split_once('/') {
                    (Some(p.trim().to_string()), Some(m.trim().to_string()))
                } else {
                    (Some(target.to_string()), None)
                };

                let complexity_is_default = complexity_key.eq_ignore_ascii_case("default");
                let task_is_default = task_key.eq_ignore_ascii_case("default");

                let specificity = match (complexity_is_default, task_is_default) {
                    (false, false) => 0, // most specific
                    (true, false) | (false, true) => 1,
                    (true, true) => 2, // least specific
                };

                matrix_rules.push((
                    specificity,
                    ClassifierRoutingRuleConfig {
                        name: format!("matrix:{complexity_key}/{task_key}"),
                        task_type: if task_is_default {
                            None
                        } else {
                            Some(task_key.clone())
                        },
                        complexity: if complexity_is_default {
                            None
                        } else {
                            Some(complexity_key.clone())
                        },
                        persona: None,
                        domain: None,
                        provider,
                        model,
                        min_confidence: None,
                    },
                ));
            }
        }

        // Sort by specificity (more specific first), stable within each tier.
        matrix_rules.sort_by_key(|(specificity, _)| *specificity);

        let mut result = self.rules.clone();
        result.extend(matrix_rules.into_iter().map(|(_, rule)| rule));
        result
    }
}

impl Default for ClassifierRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            artifacts_path: String::new(),
            confidence_threshold: default_classifier_routing_confidence_threshold(),
            fallback_to_existing_routing: default_classifier_routing_fallback_to_existing(),
            rules: Vec::new(),
            matrix: HashMap::new(),
        }
    }
}

fn default_azure_api_version() -> String {
    "2024-08-01-preview".to_string()
}

fn default_usage_log_path() -> String {
    crate::core::usage::default_usage_log_dir_string()
}

fn default_usage_retention_days() -> u64 {
    crate::core::usage::default_usage_retention_days()
}

fn default_usage_window_hours() -> u64 {
    crate::core::usage::default_usage_window_hours()
}

fn default_provider_health_check_interval_secs() -> u64 {
    300
}

fn default_classifier_routing_confidence_threshold() -> f32 {
    0.55
}

fn default_classifier_routing_fallback_to_existing() -> bool {
    true
}

impl std::fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

pub const DEFAULT_OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

pub fn default_chat_completions_url(provider: &LlmProvider) -> Option<&'static str> {
    match provider {
        LlmProvider::Openai => Some(DEFAULT_OPENAI_CHAT_COMPLETIONS_URL),
        LlmProvider::Copilot => Some("https://api.githubcopilot.com/chat/completions"),
        LlmProvider::Xai => Some("https://api.x.ai/v1/chat/completions"),
        LlmProvider::Mistral => Some("https://api.mistral.ai/v1/chat/completions"),
        LlmProvider::Groq => Some("https://api.groq.com/openai/v1/chat/completions"),
        LlmProvider::Cerebras => Some("https://api.cerebras.ai/v1/chat/completions"),
        LlmProvider::Nebius => Some("https://api.studio.nebius.ai/v1/chat/completions"),
        LlmProvider::Siliconflow => Some("https://api.siliconflow.cn/v1/chat/completions"),
        LlmProvider::Fireworks => Some("https://api.fireworks.ai/inference/v1/chat/completions"),
        LlmProvider::Nvidia => Some("https://integrate.api.nvidia.com/v1/chat/completions"),
        LlmProvider::Chutes => Some("https://llm.chutes.ai/v1/chat/completions"),
        LlmProvider::Deepseek => Some("https://api.deepseek.com/chat/completions"),
        LlmProvider::Galadriel => Some("https://api.galadriel.com/v1/chat/completions"),
        LlmProvider::Hyperbolic => Some("https://api.hyperbolic.xyz/v1/chat/completions"),
        LlmProvider::Moonshot => Some("https://api.moonshot.cn/v1/chat/completions"),
        LlmProvider::Openrouter => Some("https://openrouter.ai/api/v1/chat/completions"),
        LlmProvider::Perplexity => Some("https://api.perplexity.ai/chat/completions"),
        LlmProvider::Together => Some("https://api.together.xyz/v1/chat/completions"),
        _ => None,
    }
}

impl From<&str> for LlmProvider {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "openai" => LlmProvider::Openai,
            "azure" => LlmProvider::Azure,
            "anthropic" => LlmProvider::Anthropic,
            "copilot" => LlmProvider::Copilot,
            "xai" => LlmProvider::Xai,
            "gemini" => LlmProvider::Gemini,
            "mistral" => LlmProvider::Mistral,
            "groq" => LlmProvider::Groq,
            "cerebras" => LlmProvider::Cerebras,
            "nebius" => LlmProvider::Nebius,
            "siliconflow" => LlmProvider::Siliconflow,
            "fireworks" => LlmProvider::Fireworks,
            "nvidia" => LlmProvider::Nvidia,
            "chutes" => LlmProvider::Chutes,
            "deepseek" => LlmProvider::Deepseek,
            "cohere" => LlmProvider::Cohere,
            "galadriel" => LlmProvider::Galadriel,
            "hyperbolic" => LlmProvider::Hyperbolic,
            "huggingface" => LlmProvider::Huggingface,
            "mira" => LlmProvider::Mira,
            "moonshot" => LlmProvider::Moonshot,
            "ollama" => LlmProvider::Ollama,
            "openrouter" => LlmProvider::Openrouter,
            "perplexity" => LlmProvider::Perplexity,
            "together" => LlmProvider::Together,
            _ => LlmProvider::Openai, // default fallback
        }
    }
}

/// Cache backend for Layer 1a exact-match cache.
///
/// Set via `ISARTOR__CACHE_BACKEND` env var.
///
/// * `"memory"` — In-process LRU cache (ahash + parking_lot). Default.
/// * `"redis"`  — Distributed Redis cache for multi-replica K8s deployments.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheBackend {
    #[default]
    Memory,
    Redis,
}

/// Router backend for Layer 2 SLM intent classification.
///
/// Set via `ISARTOR__ROUTER_BACKEND` env var.
///
/// * `"embedded"` — In-process Candle inference (GGUF model). Default.
/// * `"vllm"`     — Remote vLLM / TGI inference server over HTTP.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum RouterBackend {
    #[default]
    Embedded,
    Vllm,
}

/// Classifier mode for the Layer 2 SLM triage.
///
/// Set via `ISARTOR__LAYER2__CLASSIFIER_MODE` env var.
///
/// * `"binary"` — Original SIMPLE/COMPLEX binary classification.
/// * `"tiered"` — Three-tier TEMPLATE/SNIPPET/COMPLEX classification
///   that deflects config files, type definitions, documentation, and
///   short single-function code to L2. Default.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierMode {
    Binary,
    #[default]
    Tiered,
}

// ═════════════════════════════════════════════════════════════════════
// Layer 2 Settings — Lightweight Sidecar (llama.cpp)
// ═════════════════════════════════════════════════════════════════════

/// Configuration for the Layer 2 SLM sidecar (llama.cpp server).
///
/// The sidecar exposes an **OpenAI-compatible API** at the given URL
/// and hosts a quantised SLM such as Phi-3-mini-4k-instruct-q4.gguf.
///
/// Loaded from environment variables prefixed with `ISARTOR_LAYER2__`
/// (double-underscore maps to nested struct via the `config` crate).
#[derive(Debug, Deserialize, Clone)]
pub struct Layer2Settings {
    /// Base URL of the llama.cpp sidecar (e.g. "http://127.0.0.1:8081").
    pub sidecar_url: String,

    /// Model name passed in the `"model"` field of OpenAI-compatible
    /// requests. This is informational for llama.cpp — it always uses
    /// the loaded model — but is required for API spec compliance.
    pub model_name: String,

    /// HTTP request timeout for sidecar calls, in seconds.
    pub timeout_seconds: u64,

    /// Classifier mode: `"binary"` (SIMPLE/COMPLEX) or `"tiered"`
    /// (TEMPLATE/SNIPPET/COMPLEX). Default is `"tiered"`.
    #[serde(default)]
    pub classifier_mode: ClassifierMode,

    /// Maximum tokens the SLM may generate for an L2 answer.
    /// If the SLM exceeds this, the answer is still returned but a
    /// warning is logged. Default is 2048.
    #[serde(default = "default_max_answer_tokens")]
    pub max_answer_tokens: u32,
}

fn default_max_answer_tokens() -> u32 {
    2048
}

// ═════════════════════════════════════════════════════════════════════
// Embedding Sidecar Settings — Lightweight Sidecar (llama.cpp --embedding)
// ═════════════════════════════════════════════════════════════════════

/// Configuration for the embedding sidecar (llama.cpp server with `--embedding`).
///
/// This is a separate llama.cpp instance dedicated to embedding generation,
/// running a model such as all-MiniLM-L6-v2 in GGUF format.
///
/// Loaded from environment variables prefixed with `ISARTOR_EMBEDDING_SIDECAR__`.
#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingSidecarSettings {
    /// Base URL of the embedding sidecar (e.g. "http://127.0.0.1:8082").
    pub sidecar_url: String,

    /// Model name passed in the `"model"` field of the embeddings request.
    pub model_name: String,

    /// HTTP request timeout for embedding calls, in seconds.
    pub timeout_seconds: u64,
}

/// Application configuration loaded from environment variables / config files.
#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    /// Socket address the server will bind to (e.g. "0.0.0.0:8080").
    pub host_port: String,

    /// Inference engine mode (`sidecar` or `embedded`). Default is `sidecar`.
    pub inference_engine: InferenceEngineMode,

    /// Optional API key that clients must present in the `X-API-Key` header (Layer 0).
    /// When empty, authentication is disabled (local-first default).
    pub gateway_api_key: String,

    // ── Layer 1 — Cache ─────────────────────────────────────────────
    /// Cache strategy: "exact", "semantic", or "both".
    pub cache_mode: CacheMode,

    /// Cache backend: "memory" (in-process LRU) or "redis" (distributed).
    /// Controls which `ExactCache` adapter is instantiated at startup.
    pub cache_backend: CacheBackend,

    /// Redis URL for the distributed exact-match cache.
    /// Only used when `cache_backend` = `"redis"`.
    pub redis_url: String,

    /// Router backend: "embedded" (Candle in-process) or "vllm" (remote HTTP).
    /// Controls which `SlmRouter` adapter is instantiated at startup.
    pub router_backend: RouterBackend,

    /// Base URL of the vLLM / TGI inference server.
    /// Only used when `router_backend` = `"vllm"`.
    pub vllm_url: String,

    /// Model name for the vLLM server.
    /// Only used when `router_backend` = `"vllm"`.
    pub vllm_model: String,

    /// Embedding model name (e.g. "all-minilm").
    /// Only used when `cache_mode` is `semantic` or `both`.
    pub embedding_model: String,

    /// Cosine similarity threshold for semantic cache hits (0.0–1.0).
    /// Only used when `cache_mode` is `semantic` or `both`.
    pub similarity_threshold: f64,

    /// Time-to-live for cached prompt responses, in seconds.
    pub cache_ttl_secs: u64,

    /// Maximum number of entries each cache will hold.
    pub cache_max_capacity: u64,

    // ── Layer 2 — SLM Sidecar (llama.cpp) ───────────────────────────
    /// Nested Layer 2 sidecar settings (generation model).
    pub layer2: Layer2Settings,

    // ── Legacy Layer 2 — kept for v1 middleware backwards compat ─────
    /// URL of the on-premise SLM used for intent triage (Layer 2 v1 middleware).
    /// Example: "http://localhost:11434/api/generate"
    pub local_slm_url: String,

    /// Model name to request from the local SLM (e.g. "llama3").
    pub local_slm_model: String,

    // ── Embedding Sidecar ───────────────────────────────────────────
    /// Nested embedding sidecar settings (dedicated embedding model).
    pub embedding_sidecar: EmbeddingSidecarSettings,

    // ── Layer 3 — External LLM ──────────────────────────────────────
    /// LLM provider. Supported values (all via rig-core):
    /// "openai", "azure", "anthropic", "copilot", "xai", "gemini", "mistral",
    /// "groq", "cerebras", "nebius", "siliconflow", "fireworks", "nvidia",
    /// "chutes", "deepseek", "cohere", "galadriel", "hyperbolic",
    /// "huggingface", "mira", "moonshot", "ollama", "openrouter",
    /// "perplexity", "together".
    /// Any unsupported value will cause configuration loading to fail
    /// instead of silently falling back to "openai".
    pub llm_provider: LlmProvider,

    /// Base URL for the external LLM HTTP endpoint.
    ///
    /// When `llm_provider` is `"azure"`, this value is passed as the Azure
    /// endpoint (e.g. via `azure_endpoint(...)`).
    ///
    /// For other providers, the `rig-core` client currently uses its own
    /// built-in default endpoints and ignores this setting. The following
    /// URLs are provided for documentation/reference only and may not be
    /// affected by changing `external_llm_url`:
    ///
    ///   - Azure:       https://<resource>.openai.azure.com
    ///   - Anthropic:   https://api.anthropic.com/v1/messages
    ///   - xAI:         https://api.x.ai/v1/chat/completions
    ///   - Gemini:      https://generativelanguage.googleapis.com
    ///   - Mistral:     https://api.mistral.ai/v1/chat/completions
    ///   - Groq:        https://api.groq.com/openai/v1
    ///   - DeepSeek:    https://api.deepseek.com
    ///   - Cohere:      https://api.cohere.ai
    ///   - Galadriel:   https://api.galadriel.com
    ///   - Hyperbolic:  https://api.hyperbolic.xyz/v1
    ///   - HuggingFace: https://api-inference.huggingface.co
    ///   - Mira:        https://api.mira.network
    ///   - Moonshot:    https://api.moonshot.cn/v1
    ///   - Ollama:      http://localhost:11434 (local, no API key needed)
    ///   - OpenRouter:  https://openrouter.ai/api/v1
    ///   - Perplexity:  https://api.perplexity.ai
    ///   - Together:    https://api.together.xyz
    pub external_llm_url: String,

    /// Model name to request from the external LLM.
    pub external_llm_model: String,

    /// Optional user-defined aliases that resolve short names to real provider
    /// model identifiers before routing and cache-key generation.
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,

    /// API key for the external heavy LLM (Layer 3).
    pub external_llm_api_key: String,

    /// Optional multi-account key pool for the primary provider.
    #[serde(default)]
    pub provider_keys: Vec<ProviderKeyConfig>,

    /// Strategy used when multiple primary-provider keys are configured.
    #[serde(default)]
    pub key_rotation_strategy: KeyRotationStrategy,

    /// Cooldown duration in seconds after a primary-provider key hits a
    /// rate-limit / quota-style upstream response.
    #[serde(default = "default_key_cooldown_secs")]
    pub key_cooldown_secs: u64,

    /// Ordered fallback providers used when the primary provider exhausts its
    /// retry budget with a provider-side failure such as 429/5xx/timeout.
    #[serde(default)]
    pub fallback_providers: Vec<FallbackProviderConfig>,

    /// HTTP request timeout for Layer 3 provider calls, in seconds.
    pub l3_timeout_secs: u64,

    // ── Azure-specific ──────────────────────────────────────────────
    /// Azure OpenAI deployment ID (only used when `llm_provider` = "azure").
    pub azure_deployment_id: String,

    /// Azure OpenAI API version (only used when `llm_provider` = "azure").
    pub azure_api_version: String,

    /// Optional MiniLM-based multi-head classifier and routing policy.
    #[serde(default)]
    pub classifier_routing: ClassifierRoutingConfig,

    // ── Layer 2 Feature Flag ────────────────────────────────────────
    /// Enable the Layer 2 SLM triage router (Qwen / llama.cpp sidecar).
    ///
    /// When `false` (the default), every request skips L2 entirely and
    /// goes straight from L1 cache to L3 external LLM.  Set to `true`
    /// via `ISARTOR__ENABLE_SLM_ROUTER=true` when a GPU-backed sidecar
    /// is available.
    pub enable_slm_router: bool,

    // ── L2.5 Context Optimizer ──────────────────────────────────────
    /// Enable the L2.5 context optimizer that compresses instruction
    /// payloads (CLAUDE.md, copilot-instructions.md, etc.) before
    /// forwarding to L3.  Reduces cloud input tokens in agentic mode.
    ///
    /// Set via `ISARTOR__ENABLE_CONTEXT_OPTIMIZER=true`. Default is true.
    pub enable_context_optimizer: bool,

    /// Enable cross-turn instruction deduplication within a session.
    /// When the same instruction hash is seen again, it is replaced
    /// with a compact reference.
    ///
    /// Set via `ISARTOR__CONTEXT_OPTIMIZER_DEDUP=true`. Default is true.
    pub context_optimizer_dedup: bool,

    /// Enable static minification of instruction text (strip comments,
    /// collapse whitespace, remove decorative rules).
    ///
    /// Set via `ISARTOR__CONTEXT_OPTIMIZER_MINIFY=true`. Default is true.
    pub context_optimizer_minify: bool,

    // ── Observability ───────────────────────────────────────────────
    pub enable_monitoring: bool,
    pub otel_exporter_endpoint: String,
    pub enable_request_logs: bool,
    pub request_log_path: String,
    #[serde(default = "default_usage_log_path")]
    pub usage_log_path: String,
    #[serde(default = "default_usage_retention_days")]
    pub usage_retention_days: u64,
    #[serde(default = "default_usage_window_hours")]
    pub usage_window_hours: u64,
    #[serde(default = "default_provider_health_check_interval_secs")]
    pub provider_health_check_interval_secs: u64,
    #[serde(default)]
    pub usage_pricing: HashMap<String, ProviderPricingConfig>,
    #[serde(default)]
    pub quota: HashMap<String, ProviderQuotaConfig>,

    // ── Air-Gap / Offline Mode ──────────────────────────────────────
    /// When `true`, all outbound HTTP connections are blocked at the
    /// application level and L3 Cloud Logic is disabled.
    ///
    /// Set via `ISARTOR__OFFLINE_MODE=true` or the `--offline` CLI flag.
    pub offline_mode: bool,

    // ── CONNECT Proxy ───────────────────────────────────────────────
    /// Socket address the CONNECT proxy will bind to (e.g. "0.0.0.0:8081").
    /// Used by `isartor connect copilot` to intercept Copilot CLI traffic.
    pub proxy_port: String,
}

impl AppConfig {
    /// Build configuration from environment variables prefixed with `ISARTOR_`
    /// (e.g. `ISARTOR_HOST_PORT`, `ISARTOR_GATEWAY_API_KEY`, …).
    ///
    /// Sensible defaults are provided so the binary can start without a config
    /// file during local development.
    pub fn load() -> anyhow::Result<Self> {
        Self::load_with_validation(true)
    }

    pub fn configured_model_id(&self) -> String {
        match self.llm_provider {
            LlmProvider::Azure if !self.azure_deployment_id.is_empty() => {
                self.azure_deployment_id.clone()
            }
            _ => self.external_llm_model.clone(),
        }
    }

    pub fn resolve_model_alias(&self, requested_model: &str) -> String {
        self.model_aliases
            .get(requested_model)
            .cloned()
            .unwrap_or_else(|| requested_model.to_string())
    }

    pub fn primary_provider_keys(&self) -> Vec<ProviderKeyConfig> {
        effective_provider_keys(&self.external_llm_api_key, &self.provider_keys)
    }

    pub fn has_primary_provider_key(&self) -> bool {
        !provider_requires_api_key(&self.llm_provider) || !self.primary_provider_keys().is_empty()
    }

    pub fn primary_provider_api_key(&self) -> String {
        self.primary_provider_keys()
            .first()
            .map(|entry| entry.key.clone())
            .unwrap_or_else(|| self.external_llm_api_key.clone())
    }

    pub fn quota_for_provider(&self, provider: &str) -> Option<&ProviderQuotaConfig> {
        self.quota
            .get(provider)
            .or_else(|| self.quota.get(&provider.to_lowercase()))
    }

    /// Load configuration but optionally skip strict provider validation.
    ///
    /// This is useful for startup/help flows that only need gateway-local
    /// settings (such as bind address or auth key) and should not fail just
    /// because an unused Layer 3 provider configuration is stale.
    pub fn load_with_validation(validate_provider: bool) -> anyhow::Result<Self> {
        let cfg = config::Config::builder()
            // Defaults -------------------------------------------------
            .set_default("host_port", "0.0.0.0:8080")?
            .set_default("inference_engine", "sidecar")?
            .set_default("gateway_api_key", "")?
            // Layer 1
            .set_default("cache_mode", "both")?
            .set_default("cache_backend", "memory")?
            .set_default("redis_url", "redis://127.0.0.1:6379")?
            .set_default("router_backend", "embedded")?
            .set_default("vllm_url", "http://127.0.0.1:8000")?
            .set_default("vllm_model", "gemma-2-2b-it")?
            .set_default("embedding_model", "all-minilm")?
            .set_default("similarity_threshold", 0.85)?
            .set_default("cache_ttl_secs", 300_i64)?
            .set_default("cache_max_capacity", 10_000_i64)?
            // Layer 2 — llama.cpp sidecar (generation)
            .set_default("layer2.sidecar_url", "http://127.0.0.1:8081")?
            .set_default("layer2.model_name", "phi-3-mini")?
            .set_default("layer2.timeout_seconds", 30_i64)?
            .set_default("layer2.classifier_mode", "tiered")?
            .set_default("layer2.max_answer_tokens", 2048_i64)?
            // Legacy Layer 2 (v1 middleware — Ollama compat)
            .set_default("local_slm_url", "http://localhost:11434/api/generate")?
            .set_default("local_slm_model", "llama3")?
            // Embedding sidecar (llama.cpp --embedding)
            .set_default("embedding_sidecar.sidecar_url", "http://127.0.0.1:8082")?
            .set_default("embedding_sidecar.model_name", "all-minilm")?
            .set_default("embedding_sidecar.timeout_seconds", 10_i64)?
            // Layer 3
            .set_default("llm_provider", "openai")?
            .set_default("external_llm_url", DEFAULT_OPENAI_CHAT_COMPLETIONS_URL)?
            .set_default("external_llm_model", "gpt-4o-mini")?
            .set_default("external_llm_api_key", "")?
            .set_default("key_rotation_strategy", "round_robin")?
            .set_default("key_cooldown_secs", 60_i64)?
            .set_default("l3_timeout_secs", 120_i64)?
            // Azure
            .set_default("azure_deployment_id", "")?
            .set_default("azure_api_version", "2024-08-01-preview")?
            .set_default("classifier_routing.enabled", false)?
            .set_default("classifier_routing.artifacts_path", "")?
            .set_default(
                "classifier_routing.confidence_threshold",
                default_classifier_routing_confidence_threshold() as f64,
            )?
            .set_default(
                "classifier_routing.fallback_to_existing_routing",
                default_classifier_routing_fallback_to_existing(),
            )?
            // Observability
            .set_default("enable_slm_router", false)?
            .set_default("enable_context_optimizer", true)?
            .set_default("context_optimizer_dedup", true)?
            .set_default("context_optimizer_minify", true)?
            .set_default("enable_monitoring", false)?
            .set_default("otel_exporter_endpoint", "http://localhost:4317")?
            .set_default("enable_request_logs", false)?
            .set_default(
                "request_log_path",
                crate::core::request_logger::default_request_log_dir_string(),
            )?
            .set_default(
                "usage_log_path",
                crate::core::usage::default_usage_log_dir_string(),
            )?
            .set_default(
                "usage_retention_days",
                crate::core::usage::default_usage_retention_days() as i64,
            )?
            .set_default(
                "usage_window_hours",
                crate::core::usage::default_usage_window_hours() as i64,
            )?
            .set_default(
                "provider_health_check_interval_secs",
                default_provider_health_check_interval_secs() as i64,
            )?
            // Air-gap / offline mode
            .set_default("offline_mode", false)?
            // CONNECT proxy
            .set_default("proxy_port", "0.0.0.0:8081")?
            // Optional config file --------------------------------------
            .add_source(config::File::with_name("isartor").required(false))
            // Environment overrides (ISARTOR__ prefix) -----------------
            // The `config` crate strips the prefix + prefix_separator,
            // then maps the remaining `__` sequences to nested struct
            // notation.  Because `separator("__")` also becomes the
            // default `prefix_separator`, ALL env vars must use double-
            // underscore after the ISARTOR prefix:
            //   ISARTOR__LLM_PROVIDER       → llm_provider        (top-level)
            //   ISARTOR__LAYER2__SIDECAR_URL → layer2.sidecar_url  (nested)
            .add_source(config::Environment::with_prefix("ISARTOR").separator("__"))
            .build()?;

        let mut app: AppConfig = cfg.try_deserialize()?;

        // Docker-friendly secret support: allow reading sensitive values from files
        // (e.g. Docker / Compose secrets mounted under /run/secrets/*).
        apply_secret_file_overrides(&mut app)?;
        apply_provider_key_env_overrides(&mut app)?;
        apply_fallback_provider_env_overrides(&mut app)?;
        if validate_provider {
            validate_provider_config(&app)?;
        }

        Ok(app)
    }
}

fn read_secret_file_env(var_names: &[&str]) -> anyhow::Result<Option<String>> {
    for var in var_names {
        let Ok(path) = std::env::var(var) else {
            continue;
        };

        let path = path.trim();
        if path.is_empty() {
            continue;
        }

        let content = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read secret file {var}={path}: {e}"))?;
        let secret = content.trim().to_string();

        if secret.is_empty() {
            return Err(anyhow::anyhow!(
                "secret file {var}={path} is empty (after trimming whitespace)"
            ));
        }

        return Ok(Some(secret));
    }

    Ok(None)
}

fn apply_secret_file_overrides(cfg: &mut AppConfig) -> anyhow::Result<()> {
    // Prefer explicit env/config value; only fall back to *_FILE when unset.
    if cfg.external_llm_api_key.trim().is_empty()
        && let Some(secret) = read_secret_file_env(&[
            "ISARTOR__EXTERNAL_LLM_API_KEY_FILE",
            "ISARTOR_EXTERNAL_LLM_API_KEY_FILE",
        ])?
    {
        cfg.external_llm_api_key = secret;
    }

    Ok(())
}

fn apply_fallback_provider_env_overrides(cfg: &mut AppConfig) -> anyhow::Result<()> {
    for var in ["ISARTOR__FALLBACK_PROVIDERS", "ISARTOR_FALLBACK_PROVIDERS"] {
        let Ok(raw) = std::env::var(var) else {
            continue;
        };

        let raw = raw.trim();
        if raw.is_empty() {
            cfg.fallback_providers.clear();
            return Ok(());
        }

        cfg.fallback_providers = serde_json::from_str(raw).map_err(|e| {
            anyhow::anyhow!("failed to parse {var} as JSON array of fallback provider objects: {e}")
        })?;
        return Ok(());
    }

    Ok(())
}

fn apply_provider_key_env_overrides(cfg: &mut AppConfig) -> anyhow::Result<()> {
    for var in ["ISARTOR__PROVIDER_KEYS", "ISARTOR_PROVIDER_KEYS"] {
        let Ok(raw) = std::env::var(var) else {
            continue;
        };

        let raw = raw.trim();
        if raw.is_empty() {
            cfg.provider_keys.clear();
            return Ok(());
        }

        cfg.provider_keys = serde_json::from_str(raw).map_err(|e| {
            anyhow::anyhow!("failed to parse {var} as JSON array of provider key objects: {e}")
        })?;
        return Ok(());
    }

    Ok(())
}

fn provider_requires_api_key(provider: &LlmProvider) -> bool {
    !matches!(provider, LlmProvider::Ollama)
}

pub fn effective_provider_keys(
    legacy_api_key: &str,
    provider_keys: &[ProviderKeyConfig],
) -> Vec<ProviderKeyConfig> {
    let mut keys = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let trimmed_legacy = legacy_api_key.trim();
    if !trimmed_legacy.is_empty() {
        seen.insert(trimmed_legacy.to_string());
        keys.push(ProviderKeyConfig {
            key: trimmed_legacy.to_string(),
            priority: 0,
            label: "legacy-primary".to_string(),
        });
    }

    for entry in provider_keys {
        let trimmed = entry.key.trim();
        if trimmed.is_empty() || !seen.insert(trimmed.to_string()) {
            continue;
        }
        keys.push(ProviderKeyConfig {
            key: trimmed.to_string(),
            priority: entry.priority,
            label: entry.label.trim().to_string(),
        });
    }

    keys
}

fn validate_provider_key_configs(
    provider_keys: &[ProviderKeyConfig],
    context: &str,
) -> anyhow::Result<()> {
    for (index, provider_key) in provider_keys.iter().enumerate() {
        if provider_key.key.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "{context} provider_keys[{}].key must not be empty",
                index
            ));
        }
        if provider_key.priority == 0 {
            return Err(anyhow::anyhow!(
                "{context} provider_keys[{}].priority must be greater than zero",
                index
            ));
        }
    }

    Ok(())
}

struct ProviderValidationInput<'a> {
    provider: &'a LlmProvider,
    api_key: &'a str,
    provider_keys: &'a [ProviderKeyConfig],
    endpoint: &'a str,
    model: &'a str,
    azure_deployment_id: &'a str,
    azure_api_version: &'a str,
    context: &'a str,
}

fn validate_single_provider_config(input: ProviderValidationInput<'_>) -> anyhow::Result<()> {
    if input.model.trim().is_empty() {
        return Err(anyhow::anyhow!("{} model must not be empty", input.context));
    }

    validate_provider_key_configs(input.provider_keys, input.context)?;

    let effective_keys = effective_provider_keys(input.api_key, input.provider_keys);

    if provider_requires_api_key(input.provider) && effective_keys.is_empty() {
        return Err(anyhow::anyhow!(
            "{} API key is required for provider '{}' (configure external_llm_api_key or provider_keys)",
            input.context,
            input.provider.as_str()
        ));
    }

    if *input.provider == LlmProvider::Copilot && input.api_key.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "{} GitHub token is required for GitHub Copilot access",
            input.context
        ));
    }

    if *input.provider == LlmProvider::Azure {
        if input.endpoint.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "{} Azure endpoint is empty (expected: https://<resource>.openai.azure.com)",
                input.context
            ));
        }

        if input.endpoint.contains("api.openai.com") {
            return Err(anyhow::anyhow!(
                "{} Azure endpoint points to api.openai.com; use your Azure resource endpoint instead",
                input.context
            ));
        }

        if input.endpoint.contains("/openai/")
            || input.endpoint.contains("/deployments/")
            || input.endpoint.contains("chat/completions")
            || input.endpoint.contains("api-version=")
        {
            return Err(anyhow::anyhow!(
                "{} Azure endpoint looks like a full REST URL; provide only the base endpoint",
                input.context
            ));
        }

        if !(input.endpoint.contains("openai.azure.com")
            || input.endpoint.contains("openai.azure.us")
            || input.endpoint.contains("openai.azure.cn")
            || input.endpoint.contains("cognitiveservices.azure.com"))
        {
            return Err(anyhow::anyhow!(
                "{} Azure endpoint does not look like an Azure OpenAI endpoint: '{}'",
                input.context,
                input.endpoint
            ));
        }

        if input.azure_deployment_id.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "{} Azure deployment ID is required",
                input.context
            ));
        }

        if input.azure_api_version.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "{} Azure API version is required",
                input.context
            ));
        }
    }

    Ok(())
}

fn validate_provider_config(cfg: &AppConfig) -> anyhow::Result<()> {
    validate_single_provider_config(ProviderValidationInput {
        provider: &cfg.llm_provider,
        api_key: &cfg.external_llm_api_key,
        provider_keys: &cfg.provider_keys,
        endpoint: &cfg.external_llm_url,
        model: &cfg.external_llm_model,
        azure_deployment_id: &cfg.azure_deployment_id,
        azure_api_version: &cfg.azure_api_version,
        context: "primary provider",
    })?;

    for (index, provider) in cfg.fallback_providers.iter().enumerate() {
        validate_single_provider_config(ProviderValidationInput {
            provider: &provider.provider,
            api_key: &provider.api_key,
            provider_keys: &provider.provider_keys,
            endpoint: &provider.url,
            model: &provider.model,
            azure_deployment_id: &provider.azure_deployment_id,
            azure_api_version: &provider.azure_api_version,
            context: &format!("fallback provider #{}", index + 1),
        })?;
    }

    for (alias, target) in &cfg.model_aliases {
        if alias.trim().is_empty() {
            return Err(anyhow::anyhow!("model alias names must not be empty"));
        }
        if target.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "model alias '{alias}' must resolve to a non-empty model identifier"
            ));
        }
    }

    if cfg.enable_request_logs && cfg.request_log_path.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "request logging is enabled but request_log_path is empty"
        ));
    }

    if cfg.key_cooldown_secs == 0 {
        return Err(anyhow::anyhow!(
            "key_cooldown_secs must be greater than zero when key rotation is enabled"
        ));
    }

    for (provider, quota) in &cfg.quota {
        if !(0.0..1.0).contains(&quota.warning_threshold_ratio) {
            return Err(anyhow::anyhow!(
                "quota.{provider}.warning_threshold_ratio must be between 0.0 and 1.0"
            ));
        }

        for (field, limit) in [
            ("daily_token_limit", quota.daily_token_limit),
            ("weekly_token_limit", quota.weekly_token_limit),
            ("monthly_token_limit", quota.monthly_token_limit),
        ] {
            if limit.is_some_and(|value| value == 0) {
                return Err(anyhow::anyhow!(
                    "quota.{provider}.{field} must be greater than zero"
                ));
            }
        }

        for (field, limit) in [
            ("daily_cost_limit_usd", quota.daily_cost_limit_usd),
            ("weekly_cost_limit_usd", quota.weekly_cost_limit_usd),
            ("monthly_cost_limit_usd", quota.monthly_cost_limit_usd),
        ] {
            if limit.is_some_and(|value| value <= 0.0) {
                return Err(anyhow::anyhow!(
                    "quota.{provider}.{field} must be greater than zero"
                ));
            }
        }
    }

    for (index, provider) in cfg.fallback_providers.iter().enumerate() {
        if provider.key_cooldown_secs == 0 {
            return Err(anyhow::anyhow!(
                "fallback provider #{} key_cooldown_secs must be greater than zero",
                index + 1
            ));
        }
    }

    Ok(())
}

#[cfg(any(test, feature = "test-helpers"))]
impl AppConfig {
    /// Returns a complete `AppConfig` suitable for unit/integration tests.
    ///
    /// Every field gets a sensible test value. Tests that need to override
    /// one or two fields can clone this and mutate:
    ///
    /// ```ignore
    /// let mut cfg = AppConfig::test_default();
    /// cfg.cache_mode = CacheMode::Semantic;
    /// ```
    pub fn test_default() -> Self {
        Self {
            host_port: "127.0.0.1:0".into(),
            inference_engine: InferenceEngineMode::Sidecar,
            gateway_api_key: "test-key".into(),
            cache_mode: CacheMode::Exact,
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
                sidecar_url: "http://127.0.0.1:8081".into(),
                model_name: "phi-3-mini".into(),
                timeout_seconds: 5,
                classifier_mode: ClassifierMode::Tiered,
                max_answer_tokens: 2048,
            },
            local_slm_url: "http://localhost:11434/api/generate".into(),
            local_slm_model: "llama3".into(),
            embedding_sidecar: EmbeddingSidecarSettings {
                sidecar_url: "http://127.0.0.1:8082".into(),
                model_name: "all-minilm".into(),
                timeout_seconds: 5,
            },
            llm_provider: LlmProvider::Openai,
            external_llm_url: "http://localhost".into(),
            external_llm_model: "gpt-4o-mini".into(),
            model_aliases: HashMap::new(),
            external_llm_api_key: "".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            fallback_providers: Vec::new(),
            l3_timeout_secs: 120,
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            classifier_routing: ClassifierRoutingConfig::default(),
            enable_monitoring: false,
            enable_slm_router: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            enable_request_logs: false,
            request_log_path: "~/.isartor/request_logs".into(),
            usage_log_path: "~/.isartor".into(),
            usage_retention_days: 30,
            usage_window_hours: 24,
            provider_health_check_interval_secs: default_provider_health_check_interval_secs(),
            usage_pricing: HashMap::new(),
            quota: HashMap::new(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
            enable_context_optimizer: true,
            context_optimizer_dedup: true,
            context_optimizer_minify: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_mode_default_is_both() {
        assert_eq!(CacheMode::default(), CacheMode::Both);
    }

    #[test]
    fn cache_mode_deserialize_exact() {
        let mode: CacheMode = serde_json::from_str("\"exact\"").unwrap();
        assert_eq!(mode, CacheMode::Exact);
    }

    #[test]
    fn cache_mode_deserialize_semantic() {
        let mode: CacheMode = serde_json::from_str("\"semantic\"").unwrap();
        assert_eq!(mode, CacheMode::Semantic);
    }

    #[test]
    fn cache_mode_deserialize_both() {
        let mode: CacheMode = serde_json::from_str("\"both\"").unwrap();
        assert_eq!(mode, CacheMode::Both);
    }

    #[test]
    fn cache_mode_deserialize_invalid() {
        let result = serde_json::from_str::<CacheMode>("\"unknown\"");
        assert!(result.is_err());
    }

    #[test]
    fn layer2_settings_deserialize() {
        let json =
            r#"{"sidecar_url":"http://localhost:8081","model_name":"phi-3","timeout_seconds":30}"#;
        let settings: Layer2Settings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.sidecar_url, "http://localhost:8081");
        assert_eq!(settings.model_name, "phi-3");
        assert_eq!(settings.timeout_seconds, 30);
        assert_eq!(settings.classifier_mode, ClassifierMode::Tiered);
        assert_eq!(settings.max_answer_tokens, 2048);
    }

    #[test]
    fn layer2_settings_deserialize_binary_mode() {
        let json = r#"{"sidecar_url":"http://localhost:8081","model_name":"phi-3","timeout_seconds":30,"classifier_mode":"binary","max_answer_tokens":512}"#;
        let settings: Layer2Settings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.classifier_mode, ClassifierMode::Binary);
        assert_eq!(settings.max_answer_tokens, 512);
    }

    #[test]
    fn embedding_sidecar_settings_deserialize() {
        let json = r#"{"sidecar_url":"http://localhost:8082","model_name":"all-minilm","timeout_seconds":10}"#;
        let settings: EmbeddingSidecarSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.sidecar_url, "http://localhost:8082");
        assert_eq!(settings.model_name, "all-minilm");
        assert_eq!(settings.timeout_seconds, 10);
    }

    #[test]
    fn model_aliases_deserialize_and_resolve() {
        let json = r#"{
            "host_port":"0.0.0.0:8080",
            "inference_engine":"sidecar",
            "gateway_api_key":"",
            "cache_mode":"both",
            "cache_backend":"memory",
            "redis_url":"redis://127.0.0.1:6379",
            "router_backend":"embedded",
            "vllm_url":"http://127.0.0.1:8000",
            "vllm_model":"gemma-2-2b-it",
            "embedding_model":"all-minilm",
            "similarity_threshold":0.85,
            "cache_ttl_secs":300,
            "cache_max_capacity":1000,
            "layer2":{"sidecar_url":"http://127.0.0.1:8081","model_name":"phi-3-mini","timeout_seconds":30},
            "local_slm_url":"http://localhost:11434/api/generate",
            "local_slm_model":"llama3",
            "embedding_sidecar":{"sidecar_url":"http://127.0.0.1:8082","model_name":"all-minilm","timeout_seconds":10},
            "llm_provider":"openai",
            "external_llm_url":"https://api.openai.com/v1/chat/completions",
            "external_llm_model":"gpt-4o-mini",
            "external_llm_api_key":"",
            "l3_timeout_secs":120,
            "azure_deployment_id":"",
            "azure_api_version":"2024-08-01-preview",
            "classifier_routing":{"enabled":false,"artifacts_path":"","confidence_threshold":0.55,"fallback_to_existing_routing":true,"rules":[],"matrix":{}},
            "enable_slm_router":false,
            "enable_context_optimizer":true,
            "context_optimizer_dedup":true,
            "context_optimizer_minify":true,
             "enable_monitoring":false,
             "otel_exporter_endpoint":"http://localhost:4317",
             "enable_request_logs":false,
             "request_log_path":"~/.isartor/request_logs",
             "provider_health_check_interval_secs":300,
             "offline_mode":false,
             "proxy_port":"0.0.0.0:8081",
             "model_aliases":{"fast":"gpt-4o-mini","smart":"gpt-4o"}
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.resolve_model_alias("fast"), "gpt-4o-mini");
        assert_eq!(config.resolve_model_alias("smart"), "gpt-4o");
        assert_eq!(config.resolve_model_alias("raw-model"), "raw-model");
    }

    #[test]
    fn default_chat_completion_urls_cover_new_openai_compatible_providers() {
        assert_eq!(
            default_chat_completions_url(&LlmProvider::Cerebras),
            Some("https://api.cerebras.ai/v1/chat/completions")
        );
        assert_eq!(
            default_chat_completions_url(&LlmProvider::Nebius),
            Some("https://api.studio.nebius.ai/v1/chat/completions")
        );
        assert_eq!(
            default_chat_completions_url(&LlmProvider::Siliconflow),
            Some("https://api.siliconflow.cn/v1/chat/completions")
        );
        assert_eq!(
            default_chat_completions_url(&LlmProvider::Fireworks),
            Some("https://api.fireworks.ai/inference/v1/chat/completions")
        );
        assert_eq!(
            default_chat_completions_url(&LlmProvider::Nvidia),
            Some("https://integrate.api.nvidia.com/v1/chat/completions")
        );
        assert_eq!(
            default_chat_completions_url(&LlmProvider::Chutes),
            Some("https://llm.chutes.ai/v1/chat/completions")
        );
    }

    #[test]
    fn app_config_loads_with_defaults() {
        let cfg = config::Config::builder()
            .set_default("host_port", "0.0.0.0:8080")
            .unwrap()
            .set_default("inference_engine", "sidecar")
            .unwrap()
            .set_default("gateway_api_key", "")
            .unwrap()
            .set_default("cache_mode", "both")
            .unwrap()
            .set_default("cache_backend", "memory")
            .unwrap()
            .set_default("redis_url", "redis://127.0.0.1:6379")
            .unwrap()
            .set_default("router_backend", "embedded")
            .unwrap()
            .set_default("vllm_url", "http://127.0.0.1:8000")
            .unwrap()
            .set_default("vllm_model", "gemma-2-2b-it")
            .unwrap()
            .set_default("embedding_model", "all-minilm")
            .unwrap()
            .set_default("similarity_threshold", 0.85)
            .unwrap()
            .set_default("cache_ttl_secs", 300_i64)
            .unwrap()
            .set_default("cache_max_capacity", 10_000_i64)
            .unwrap()
            .set_default("layer2.sidecar_url", "http://127.0.0.1:8081")
            .unwrap()
            .set_default("layer2.model_name", "phi-3-mini")
            .unwrap()
            .set_default("layer2.timeout_seconds", 30_i64)
            .unwrap()
            .set_default("layer2.classifier_mode", "tiered")
            .unwrap()
            .set_default("layer2.max_answer_tokens", 2048_i64)
            .unwrap()
            .set_default("local_slm_url", "http://localhost:11434/api/generate")
            .unwrap()
            .set_default("local_slm_model", "llama3")
            .unwrap()
            .set_default("embedding_sidecar.sidecar_url", "http://127.0.0.1:8082")
            .unwrap()
            .set_default("embedding_sidecar.model_name", "all-minilm")
            .unwrap()
            .set_default("embedding_sidecar.timeout_seconds", 10_i64)
            .unwrap()
            .set_default("llm_provider", "openai")
            .unwrap()
            .set_default(
                "external_llm_url",
                "https://api.openai.com/v1/chat/completions",
            )
            .unwrap()
            .set_default("external_llm_model", "gpt-4o-mini")
            .unwrap()
            .set_default("external_llm_api_key", "")
            .unwrap()
            .set_default("l3_timeout_secs", 120_i64)
            .unwrap()
            .set_default("azure_deployment_id", "")
            .unwrap()
            .set_default("azure_api_version", "2024-08-01-preview")
            .unwrap()
            .set_default("enable_monitoring", false)
            .unwrap()
            .set_default("enable_slm_router", false)
            .unwrap()
            .set_default("enable_context_optimizer", true)
            .unwrap()
            .set_default("context_optimizer_dedup", true)
            .unwrap()
            .set_default("context_optimizer_minify", true)
            .unwrap()
            .set_default("otel_exporter_endpoint", "http://localhost:4317")
            .unwrap()
            .set_default("enable_request_logs", false)
            .unwrap()
            .set_default("request_log_path", "~/.isartor/request_logs")
            .unwrap()
            .set_default("offline_mode", false)
            .unwrap()
            .set_default("proxy_port", "0.0.0.0:8081")
            .unwrap()
            .build()
            .unwrap();

        let config: AppConfig = cfg.try_deserialize().unwrap();

        assert_eq!(config.host_port, "0.0.0.0:8080");
        assert_eq!(config.inference_engine, InferenceEngineMode::Sidecar);
        assert_eq!(config.gateway_api_key, "");
        assert_eq!(config.cache_mode, CacheMode::Both);
        assert_eq!(config.cache_backend, CacheBackend::Memory);
        assert_eq!(config.redis_url, "redis://127.0.0.1:6379");
        assert_eq!(config.router_backend, RouterBackend::Embedded);
        assert_eq!(config.vllm_url, "http://127.0.0.1:8000");
        assert_eq!(config.vllm_model, "gemma-2-2b-it");
        assert_eq!(config.embedding_model, "all-minilm");
        assert!((config.similarity_threshold - 0.85).abs() < 1e-9);
        assert_eq!(config.cache_ttl_secs, 300);
        assert_eq!(config.cache_max_capacity, 10_000);
        assert_eq!(config.layer2.sidecar_url, "http://127.0.0.1:8081");
        assert_eq!(config.layer2.model_name, "phi-3-mini");
        assert_eq!(config.layer2.timeout_seconds, 30);
        assert_eq!(config.layer2.classifier_mode, ClassifierMode::Tiered);
        assert_eq!(config.layer2.max_answer_tokens, 2048);
        assert_eq!(
            config.embedding_sidecar.sidecar_url,
            "http://127.0.0.1:8082"
        );
        assert_eq!(config.embedding_sidecar.model_name, "all-minilm");
        assert_eq!(config.embedding_sidecar.timeout_seconds, 10);
        assert_eq!(config.llm_provider, "openai".into());
        assert_eq!(config.external_llm_model, "gpt-4o-mini");
        assert_eq!(config.l3_timeout_secs, 120);
        assert!(!config.enable_monitoring);
        assert!(!config.enable_request_logs);
        assert_eq!(config.request_log_path, "~/.isartor/request_logs");
        assert!(!config.enable_slm_router);
        assert!(config.enable_context_optimizer);
        assert!(config.context_optimizer_dedup);
        assert!(config.context_optimizer_minify);
    }

    #[test]
    fn app_config_env_var_override() {
        // Build config directly from the builder with env overrides injected
        // as explicit config values, avoiding env::set_var race conditions.
        let cfg = config::Config::builder()
            .set_default("host_port", "0.0.0.0:8080")
            .unwrap()
            .set_default("gateway_api_key", "")
            .unwrap()
            .set_default("cache_mode", "both")
            .unwrap()
            .set_default("cache_backend", "memory")
            .unwrap()
            .set_default("redis_url", "redis://127.0.0.1:6379")
            .unwrap()
            .set_default("router_backend", "embedded")
            .unwrap()
            .set_default("vllm_url", "http://127.0.0.1:8000")
            .unwrap()
            .set_default("vllm_model", "gemma-2-2b-it")
            .unwrap()
            .set_default("embedding_model", "all-minilm")
            .unwrap()
            .set_default("similarity_threshold", 0.85)
            .unwrap()
            .set_default("cache_ttl_secs", 300_i64)
            .unwrap()
            .set_default("cache_max_capacity", 10_000_i64)
            .unwrap()
            .set_default("layer2.sidecar_url", "http://127.0.0.1:8081")
            .unwrap()
            .set_default("layer2.model_name", "phi-3-mini")
            .unwrap()
            .set_default("layer2.timeout_seconds", 30_i64)
            .unwrap()
            .set_default("layer2.classifier_mode", "tiered")
            .unwrap()
            .set_default("layer2.max_answer_tokens", 2048_i64)
            .unwrap()
            .set_default("local_slm_url", "http://localhost:11434/api/generate")
            .unwrap()
            .set_default("local_slm_model", "llama3")
            .unwrap()
            .set_default("embedding_sidecar.sidecar_url", "http://127.0.0.1:8082")
            .unwrap()
            .set_default("embedding_sidecar.model_name", "all-minilm")
            .unwrap()
            .set_default("embedding_sidecar.timeout_seconds", 10_i64)
            .unwrap()
            .set_default("llm_provider", "openai")
            .unwrap()
            .set_default(
                "external_llm_url",
                "https://api.openai.com/v1/chat/completions",
            )
            .unwrap()
            .set_default("external_llm_model", "gpt-4o-mini")
            .unwrap()
            .set_default("external_llm_api_key", "")
            .unwrap()
            .set_default("l3_timeout_secs", 120_i64)
            .unwrap()
            .set_default("azure_deployment_id", "")
            .unwrap()
            .set_default("azure_api_version", "2024-08-01-preview")
            .unwrap()
            .set_default("enable_monitoring", false)
            .unwrap()
            .set_default("enable_slm_router", false)
            .unwrap()
            .set_default("enable_context_optimizer", true)
            .unwrap()
            .set_default("context_optimizer_dedup", true)
            .unwrap()
            .set_default("context_optimizer_minify", true)
            .unwrap()
            .set_default("otel_exporter_endpoint", "http://localhost:4317")
            .unwrap()
            .set_default("enable_request_logs", false)
            .unwrap()
            .set_default("request_log_path", "~/.isartor/request_logs")
            .unwrap()
            .set_default("offline_mode", false)
            .unwrap()
            .set_default("proxy_port", "0.0.0.0:8081")
            .unwrap()
            .set_default("inference_engine", "sidecar")
            .unwrap()
            // Simulate env overrides by setting values directly.
            .set_override("host_port", "127.0.0.1:9090")
            .unwrap()
            .set_override("gateway_api_key", "my-secret-key")
            .unwrap()
            .set_override("cache_mode", "exact")
            .unwrap()
            .set_override("cache_ttl_secs", 600_i64)
            .unwrap()
            .set_override("enable_monitoring", true)
            .unwrap()
            .set_override("enable_request_logs", true)
            .unwrap()
            .set_override("request_log_path", "/tmp/isartor-requests")
            .unwrap()
            .build()
            .unwrap();

        let config: AppConfig = cfg.try_deserialize().unwrap();

        assert_eq!(config.host_port, "127.0.0.1:9090");
        assert_eq!(config.inference_engine, InferenceEngineMode::Sidecar);
        assert_eq!(config.gateway_api_key, "my-secret-key");
        assert_eq!(config.cache_mode, CacheMode::Exact);
        assert_eq!(config.cache_ttl_secs, 600);
        assert!(config.enable_monitoring);
        assert!(config.enable_request_logs);
        assert_eq!(config.request_log_path, "/tmp/isartor-requests");
        assert!(!config.enable_slm_router);
    }

    #[test]
    fn app_config_nested_env_override() {
        // Build config directly with nested overrides to avoid env::set_var issues.
        let cfg = config::Config::builder()
            .set_default("host_port", "0.0.0.0:8080")
            .unwrap()
            .set_default("gateway_api_key", "")
            .unwrap()
            .set_default("cache_mode", "both")
            .unwrap()
            .set_default("cache_backend", "memory")
            .unwrap()
            .set_default("redis_url", "redis://127.0.0.1:6379")
            .unwrap()
            .set_default("router_backend", "embedded")
            .unwrap()
            .set_default("vllm_url", "http://127.0.0.1:8000")
            .unwrap()
            .set_default("vllm_model", "gemma-2-2b-it")
            .unwrap()
            .set_default("embedding_model", "all-minilm")
            .unwrap()
            .set_default("similarity_threshold", 0.85)
            .unwrap()
            .set_default("cache_ttl_secs", 300_i64)
            .unwrap()
            .set_default("cache_max_capacity", 10_000_i64)
            .unwrap()
            .set_default("layer2.sidecar_url", "http://127.0.0.1:8081")
            .unwrap()
            .set_default("layer2.model_name", "phi-3-mini")
            .unwrap()
            .set_default("layer2.timeout_seconds", 30_i64)
            .unwrap()
            .set_default("layer2.classifier_mode", "tiered")
            .unwrap()
            .set_default("layer2.max_answer_tokens", 2048_i64)
            .unwrap()
            .set_default("local_slm_url", "http://localhost:11434/api/generate")
            .unwrap()
            .set_default("local_slm_model", "llama3")
            .unwrap()
            .set_default("embedding_sidecar.sidecar_url", "http://127.0.0.1:8082")
            .unwrap()
            .set_default("embedding_sidecar.model_name", "all-minilm")
            .unwrap()
            .set_default("embedding_sidecar.timeout_seconds", 10_i64)
            .unwrap()
            .set_default("llm_provider", "openai")
            .unwrap()
            .set_default(
                "external_llm_url",
                "https://api.openai.com/v1/chat/completions",
            )
            .unwrap()
            .set_default("external_llm_model", "gpt-4o-mini")
            .unwrap()
            .set_default("external_llm_api_key", "")
            .unwrap()
            .set_default("l3_timeout_secs", 120_i64)
            .unwrap()
            .set_default("azure_deployment_id", "")
            .unwrap()
            .set_default("azure_api_version", "2024-08-01-preview")
            .unwrap()
            .set_default("enable_monitoring", false)
            .unwrap()
            .set_default("enable_slm_router", false)
            .unwrap()
            .set_default("enable_context_optimizer", true)
            .unwrap()
            .set_default("context_optimizer_dedup", true)
            .unwrap()
            .set_default("context_optimizer_minify", true)
            .unwrap()
            .set_default("otel_exporter_endpoint", "http://localhost:4317")
            .unwrap()
            .set_default("enable_request_logs", false)
            .unwrap()
            .set_default("request_log_path", "~/.isartor/request_logs")
            .unwrap()
            .set_default("offline_mode", false)
            .unwrap()
            .set_default("proxy_port", "0.0.0.0:8081")
            .unwrap()
            .set_override("inference_engine", "sidecar")
            .unwrap()
            // Nested struct overrides.
            .set_override("layer2.sidecar_url", "http://custom:9999")
            .unwrap()
            .set_override("layer2.model_name", "custom-model")
            .unwrap()
            .set_override("layer2.timeout_seconds", 60_i64)
            .unwrap()
            .set_override("embedding_sidecar.sidecar_url", "http://embed:7777")
            .unwrap()
            .build()
            .unwrap();

        let config: AppConfig = cfg.try_deserialize().unwrap();

        assert_eq!(config.layer2.sidecar_url, "http://custom:9999");
        assert_eq!(config.layer2.model_name, "custom-model");
        assert_eq!(config.layer2.timeout_seconds, 60);
        assert_eq!(config.embedding_sidecar.sidecar_url, "http://embed:7777");
    }

    #[test]
    fn cache_mode_clone_and_eq() {
        let mode = CacheMode::Exact;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);

        assert_ne!(CacheMode::Exact, CacheMode::Semantic);
        assert_ne!(CacheMode::Semantic, CacheMode::Both);
    }

    #[test]
    fn inference_engine_embedded_via_config_crate() {
        // Ensure the config crate can deserialize "embedded" into InferenceEngineMode::Embedded.
        let cfg = config::Config::builder()
            .set_default("host_port", "0.0.0.0:8080")
            .unwrap()
            .set_default("gateway_api_key", "")
            .unwrap()
            .set_default("cache_mode", "both")
            .unwrap()
            .set_default("cache_backend", "memory")
            .unwrap()
            .set_default("redis_url", "redis://127.0.0.1:6379")
            .unwrap()
            .set_default("router_backend", "embedded")
            .unwrap()
            .set_default("vllm_url", "http://127.0.0.1:8000")
            .unwrap()
            .set_default("vllm_model", "gemma-2-2b-it")
            .unwrap()
            .set_default("embedding_model", "all-minilm")
            .unwrap()
            .set_default("similarity_threshold", 0.85)
            .unwrap()
            .set_default("cache_ttl_secs", 300_i64)
            .unwrap()
            .set_default("cache_max_capacity", 10_000_i64)
            .unwrap()
            .set_default("layer2.sidecar_url", "http://127.0.0.1:8081")
            .unwrap()
            .set_default("layer2.model_name", "phi-3-mini")
            .unwrap()
            .set_default("layer2.timeout_seconds", 30_i64)
            .unwrap()
            .set_default("layer2.classifier_mode", "tiered")
            .unwrap()
            .set_default("layer2.max_answer_tokens", 2048_i64)
            .unwrap()
            .set_default("local_slm_url", "http://localhost:11434/api/generate")
            .unwrap()
            .set_default("local_slm_model", "llama3")
            .unwrap()
            .set_default("embedding_sidecar.sidecar_url", "http://127.0.0.1:8082")
            .unwrap()
            .set_default("embedding_sidecar.model_name", "all-minilm")
            .unwrap()
            .set_default("embedding_sidecar.timeout_seconds", 10_i64)
            .unwrap()
            .set_default("llm_provider", "openai")
            .unwrap()
            .set_default(
                "external_llm_url",
                "https://api.openai.com/v1/chat/completions",
            )
            .unwrap()
            .set_default("external_llm_model", "gpt-4o-mini")
            .unwrap()
            .set_default("external_llm_api_key", "")
            .unwrap()
            .set_default("l3_timeout_secs", 120_i64)
            .unwrap()
            .set_default("azure_deployment_id", "")
            .unwrap()
            .set_default("azure_api_version", "2024-08-01-preview")
            .unwrap()
            .set_default("enable_monitoring", false)
            .unwrap()
            .set_default("enable_slm_router", false)
            .unwrap()
            .set_default("enable_context_optimizer", true)
            .unwrap()
            .set_default("context_optimizer_dedup", true)
            .unwrap()
            .set_default("context_optimizer_minify", true)
            .unwrap()
            .set_default("otel_exporter_endpoint", "http://localhost:4317")
            .unwrap()
            .set_default("enable_request_logs", false)
            .unwrap()
            .set_default("request_log_path", "~/.isartor/request_logs")
            .unwrap()
            .set_default("offline_mode", false)
            .unwrap()
            .set_default("proxy_port", "0.0.0.0:8081")
            .unwrap()
            // Set inference_engine to "embedded" — the key test.
            .set_override("inference_engine", "embedded")
            .unwrap()
            .build()
            .unwrap();

        let config: AppConfig = cfg.try_deserialize().unwrap();
        assert_eq!(config.inference_engine, InferenceEngineMode::Embedded);
    }

    /// Verifies that `AppConfig::load()` picks up env vars with the double-
    /// underscore prefix separator (`ISARTOR__LLM_PROVIDER`) required by the
    /// config crate when `separator("__")` is used.
    #[test]
    fn env_var_double_underscore_prefix() {
        temp_env::with_vars(
            vec![
                ("ISARTOR__INFERENCE_ENGINE", Some("embedded")),
                ("ISARTOR__LLM_PROVIDER", Some("azure")),
                ("ISARTOR__EXTERNAL_LLM_API_KEY", Some("test-key-123")),
                ("ISARTOR__L3_TIMEOUT_SECS", Some("45")),
                (
                    "ISARTOR__EXTERNAL_LLM_URL",
                    Some("https://example.openai.azure.com"),
                ),
                ("ISARTOR__AZURE_DEPLOYMENT_ID", Some("my-deployment")),
                ("ISARTOR__AZURE_API_VERSION", Some("2024-08-01-preview")),
                ("ISARTOR__LAYER2__SIDECAR_URL", Some("http://custom:9999")),
            ],
            || {
                let config = AppConfig::load().expect("load must succeed");
                assert_eq!(config.inference_engine, InferenceEngineMode::Embedded);
                assert_eq!(config.llm_provider, "azure".into());
                assert_eq!(config.external_llm_api_key, "test-key-123");
                assert_eq!(config.l3_timeout_secs, 45);
                assert_eq!(config.layer2.sidecar_url, "http://custom:9999");
            },
        );
    }

    #[test]
    fn external_llm_api_key_file_is_used_when_key_empty() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let tmp = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = tmp.join(format!("isartor-secret-{}-{nanos}", std::process::id()));

        std::fs::write(&path, "file-secret-key\n").unwrap();

        let path_str = path.to_string_lossy().to_string();

        temp_env::with_vars(
            vec![
                ("ISARTOR__EXTERNAL_LLM_API_KEY", Some("")),
                (
                    "ISARTOR__EXTERNAL_LLM_API_KEY_FILE",
                    Some(path_str.as_str()),
                ),
            ],
            || {
                let config = AppConfig::load().expect("load must succeed");
                assert_eq!(config.external_llm_api_key, "file-secret-key");
            },
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn azure_provider_accepts_key_from_file() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let tmp = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = tmp.join(format!(
            "isartor-azure-secret-{}-{nanos}",
            std::process::id()
        ));

        std::fs::write(&path, "azure-secret").unwrap();
        let path_str = path.to_string_lossy().to_string();

        temp_env::with_vars(
            vec![
                ("ISARTOR__LLM_PROVIDER", Some("azure")),
                (
                    "ISARTOR__EXTERNAL_LLM_URL",
                    Some("https://example.openai.azure.com"),
                ),
                ("ISARTOR__AZURE_DEPLOYMENT_ID", Some("my-deployment")),
                ("ISARTOR__AZURE_API_VERSION", Some("2024-08-01-preview")),
                ("ISARTOR__EXTERNAL_LLM_API_KEY", Some("")),
                (
                    "ISARTOR__EXTERNAL_LLM_API_KEY_FILE",
                    Some(path_str.as_str()),
                ),
            ],
            || {
                let config = AppConfig::load().expect("load must succeed");
                assert_eq!(config.llm_provider, "azure".into());
                assert_eq!(config.external_llm_api_key, "azure-secret");
            },
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn relaxed_load_allows_stale_azure_provider_settings() {
        temp_env::with_vars(
            vec![
                ("ISARTOR__LLM_PROVIDER", Some("azure")),
                (
                    "ISARTOR__EXTERNAL_LLM_URL",
                    Some("https://api.openai.com/v1/chat/completions"),
                ),
                ("ISARTOR__EXTERNAL_LLM_API_KEY", Some("")),
                ("ISARTOR__AZURE_DEPLOYMENT_ID", Some("")),
            ],
            || {
                let strict = AppConfig::load();
                assert!(strict.is_err());

                let relaxed =
                    AppConfig::load_with_validation(false).expect("relaxed load must succeed");
                assert_eq!(relaxed.llm_provider, "azure".into());
                assert_eq!(
                    relaxed.external_llm_url,
                    "https://api.openai.com/v1/chat/completions"
                );
            },
        );
    }
}

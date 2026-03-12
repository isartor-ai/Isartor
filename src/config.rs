#![allow(dead_code)]
use serde::Deserialize;

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

    /// API key that clients must present in the `X-API-Key` header (Layer 0).
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
    /// LLM provider: "openai", "azure", "anthropic", or "xai".
    pub llm_provider: String,

    /// Base URL for the external LLM API.
    ///   - OpenAI:    https://api.openai.com/v1/chat/completions
    ///   - Azure:     https://<resource>.openai.azure.com
    ///   - Anthropic: https://api.anthropic.com/v1/messages
    ///   - xAI:       https://api.x.ai/v1/chat/completions
    pub external_llm_url: String,

    /// Model name to request from the external LLM.
    pub external_llm_model: String,

    /// API key for the external heavy LLM (Layer 3).
    pub external_llm_api_key: String,

    // ── Azure-specific ──────────────────────────────────────────────
    /// Azure OpenAI deployment ID (only used when `llm_provider` = "azure").
    pub azure_deployment_id: String,

    /// Azure OpenAI API version (only used when `llm_provider` = "azure").
    pub azure_api_version: String,

    // ── Layer 2 Feature Flag ────────────────────────────────────────
    /// Enable the Layer 2 SLM triage router (Qwen / llama.cpp sidecar).
    ///
    /// When `false` (the default), every request skips L2 entirely and
    /// goes straight from L1 cache to L3 external LLM.  Set to `true`
    /// via `ISARTOR__ENABLE_SLM_ROUTER=true` when a GPU-backed sidecar
    /// is available.
    pub enable_slm_router: bool,

    // ── Observability ───────────────────────────────────────────────
    pub enable_monitoring: bool,
    pub otel_exporter_endpoint: String,
}

impl AppConfig {
    /// Build configuration from environment variables prefixed with `ISARTOR_`
    /// (e.g. `ISARTOR_HOST_PORT`, `ISARTOR_GATEWAY_API_KEY`, …).
    ///
    /// Sensible defaults are provided so the binary can start without a config
    /// file during local development.
    pub fn load() -> anyhow::Result<Self> {
        let cfg = config::Config::builder()
            // Defaults -------------------------------------------------
            .set_default("host_port", "0.0.0.0:8080")?
            .set_default("inference_engine", "sidecar")?
            .set_default("gateway_api_key", "changeme")?
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
            // Legacy Layer 2 (v1 middleware — Ollama compat)
            .set_default("local_slm_url", "http://localhost:11434/api/generate")?
            .set_default("local_slm_model", "llama3")?
            // Embedding sidecar (llama.cpp --embedding)
            .set_default("embedding_sidecar.sidecar_url", "http://127.0.0.1:8082")?
            .set_default("embedding_sidecar.model_name", "all-minilm")?
            .set_default("embedding_sidecar.timeout_seconds", 10_i64)?
            // Layer 3
            .set_default("llm_provider", "openai")?
            .set_default(
                "external_llm_url",
                "https://api.openai.com/v1/chat/completions",
            )?
            .set_default("external_llm_model", "gpt-4o-mini")?
            .set_default("external_llm_api_key", "")?
            // Azure
            .set_default("azure_deployment_id", "")?
            .set_default("azure_api_version", "2024-08-01-preview")?
            // Observability
            .set_default("enable_slm_router", false)?
            .set_default("enable_monitoring", false)?
            .set_default("otel_exporter_endpoint", "http://localhost:4317")?
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

        Ok(cfg.try_deserialize()?)
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
    fn app_config_loads_with_defaults() {
        // Use temp-env to ensure no stale env vars leak in.
        temp_env::with_vars_unset(
            vec![
                "ISARTOR_HOST_PORT",
                "ISARTOR_GATEWAY_API_KEY",
                "ISARTOR_CACHE_MODE",
                "ISARTOR_CACHE_TTL_SECS",
                "ISARTOR_ENABLE_MONITORING",
                "ISARTOR_ENABLE_SLM_ROUTER",
                "ISARTOR_LAYER2__SIDECAR_URL",
                "ISARTOR_LAYER2__MODEL_NAME",
                "ISARTOR_LAYER2__TIMEOUT_SECONDS",
                "ISARTOR_EMBEDDING_SIDECAR__SIDECAR_URL",
                "ISARTOR_EMBEDDING_SIDECAR__MODEL_NAME",
                "ISARTOR_EMBEDDING_SIDECAR__TIMEOUT_SECONDS",
            ],
            || {
                let config = AppConfig::load().expect("default load must succeed");

                assert_eq!(config.host_port, "0.0.0.0:8080");
                assert_eq!(config.inference_engine, InferenceEngineMode::Sidecar);
                assert_eq!(config.gateway_api_key, "changeme");
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
                assert_eq!(
                    config.embedding_sidecar.sidecar_url,
                    "http://127.0.0.1:8082"
                );
                assert_eq!(config.embedding_sidecar.model_name, "all-minilm");
                assert_eq!(config.embedding_sidecar.timeout_seconds, 10);
                assert_eq!(config.llm_provider, "openai");
                assert_eq!(config.external_llm_model, "gpt-4o-mini");
                assert!(!config.enable_monitoring);
                assert!(!config.enable_slm_router);
            },
        );
    }

    #[test]
    fn app_config_env_var_override() {
        // Build config directly from the builder with env overrides injected
        // as explicit config values, avoiding env::set_var race conditions.
        let cfg = config::Config::builder()
            .set_default("host_port", "0.0.0.0:8080")
            .unwrap()
            .set_default("gateway_api_key", "changeme")
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
            .set_default("azure_deployment_id", "")
            .unwrap()
            .set_default("azure_api_version", "2024-08-01-preview")
            .unwrap()
            .set_default("enable_monitoring", false)
            .unwrap()
            .set_default("enable_slm_router", false)
            .unwrap()
            .set_default("otel_exporter_endpoint", "http://localhost:4317")
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
            .build()
            .unwrap();

        let config: AppConfig = cfg.try_deserialize().unwrap();

        assert_eq!(config.host_port, "127.0.0.1:9090");
        assert_eq!(config.inference_engine, InferenceEngineMode::Sidecar);
        assert_eq!(config.gateway_api_key, "my-secret-key");
        assert_eq!(config.cache_mode, CacheMode::Exact);
        assert_eq!(config.cache_ttl_secs, 600);
        assert!(config.enable_monitoring);
        assert!(!config.enable_slm_router);
    }

    #[test]
    fn app_config_nested_env_override() {
        // Build config directly with nested overrides to avoid env::set_var issues.
        let cfg = config::Config::builder()
            .set_default("host_port", "0.0.0.0:8080")
            .unwrap()
            .set_default("gateway_api_key", "changeme")
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
            .set_default("azure_deployment_id", "")
            .unwrap()
            .set_default("azure_api_version", "2024-08-01-preview")
            .unwrap()
            .set_default("enable_monitoring", false)
            .unwrap()
            .set_default("enable_slm_router", false)
            .unwrap()
            .set_default("otel_exporter_endpoint", "http://localhost:4317")
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
            .set_default("gateway_api_key", "changeme")
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
            .set_default("azure_deployment_id", "")
            .unwrap()
            .set_default("azure_api_version", "2024-08-01-preview")
            .unwrap()
            .set_default("enable_monitoring", false)
            .unwrap()
            .set_default("enable_slm_router", false)
            .unwrap()
            .set_default("otel_exporter_endpoint", "http://localhost:4317")
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
                ("ISARTOR__LAYER2__SIDECAR_URL", Some("http://custom:9999")),
            ],
            || {
                let config = AppConfig::load().expect("load must succeed");
                assert_eq!(config.inference_engine, InferenceEngineMode::Embedded);
                assert_eq!(config.llm_provider, "azure");
                assert_eq!(config.external_llm_api_key, "test-key-123");
                assert_eq!(config.layer2.sidecar_url, "http://custom:9999");
            },
        );
    }
}

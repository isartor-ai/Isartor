#![allow(dead_code)]
//! # Factory — Configuration-driven adapter instantiation
//!
//! Reads `AppConfig` and wires the correct **Port → Adapter** bindings
//! at startup.  This is the single place where the decision between
//! Minimalist (in-process) and Enterprise (distributed) modes is made.
//!
//! ```text
//! ┌──────────────┐
//! │  AppConfig    │
//! │ cache_backend │──► "memory" → InMemoryCache
//! │               │──► "redis"  → RedisExactCache
//! │router_backend │──► "embedded" → EmbeddedCandleRouter
//! │               │──► "vllm"     → RemoteVllmRouter
//! └──────────────┘
//! ```

use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::adapters::cache::{InMemoryCache, RedisExactCache};
use crate::adapters::router::{EmbeddedCandleRouter, RemoteVllmRouter};
use crate::config::{AppConfig, CacheBackend, RouterBackend};
use crate::core::ports::{ExactCache, SlmRouter};

/// Build the `ExactCache` adapter based on `config.cache_backend`.
///
/// | `cache_backend` | Adapter           | Notes                                   |
/// |-----------------|-------------------|-----------------------------------------|
/// | `memory`        | `InMemoryCache`   | ahash + LRU, zero external deps         |
/// | `redis`         | `RedisExactCache` | Uses `config.redis_url` for connection   |
pub fn build_exact_cache(config: &AppConfig) -> Arc<dyn ExactCache> {
    match config.cache_backend {
        CacheBackend::Memory => {
            let capacity = NonZeroUsize::new(config.cache_max_capacity as usize)
                .unwrap_or_else(|| NonZeroUsize::new(128).unwrap());
            log::info!(
                "Factory: ExactCache → InMemoryCache (capacity={})",
                capacity
            );
            Arc::new(InMemoryCache::new(capacity))
        }
        CacheBackend::Redis => {
            log::info!(
                "Factory: ExactCache → RedisExactCache (url={})",
                config.redis_url
            );
            Arc::new(RedisExactCache::new(&config.redis_url))
        }
    }
}

/// Build the `SlmRouter` adapter based on `config.router_backend`.
///
/// | `router_backend` | Adapter                | Notes                                      |
/// |-------------------|------------------------|--------------------------------------------|
/// | `embedded`        | `EmbeddedCandleRouter` | In-process Candle GGUF inference            |
/// | `vllm`            | `RemoteVllmRouter`     | HTTP to vLLM/TGI at `config.vllm_url`      |
pub fn build_slm_router(config: &AppConfig, http_client: &reqwest::Client) -> Arc<dyn SlmRouter> {
    match config.router_backend {
        RouterBackend::Embedded => {
            log::info!("Factory: SlmRouter → EmbeddedCandleRouter");
            Arc::new(EmbeddedCandleRouter::new(
                "mradermacher/gemma-2-2b-it-GGUF",
                "gemma-2-2b-it.Q4_K_M.gguf",
            ))
        }
        RouterBackend::Vllm => {
            log::info!(
                "Factory: SlmRouter → RemoteVllmRouter (url={}, model={})",
                config.vllm_url,
                config.vllm_model
            );
            Arc::new(RemoteVllmRouter::new(
                http_client.clone(),
                &config.vllm_url,
                &config.vllm_model,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn minimal_config() -> AppConfig {
        AppConfig {
            host_port: "127.0.0.1:0".into(),
            inference_engine: InferenceEngineMode::Sidecar,
            gateway_api_key: "test".into(),
            cache_mode: CacheMode::Both,
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
                model_name: "test".into(),
                timeout_seconds: 5,
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
            external_llm_model: "test".into(),
            external_llm_api_key: "".into(),
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
            enable_slm_router: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            offline_mode: false,
        }
    }

    #[tokio::test]
    async fn factory_builds_in_memory_cache() {
        let config = minimal_config();
        let cache = build_exact_cache(&config);
        // Should start empty.
        assert!(cache.get("key").await.unwrap().is_none());
        cache.put("key", "value").await.unwrap();
        assert_eq!(cache.get("key").await.unwrap(), Some("value".into()));
    }

    #[tokio::test]
    #[ignore = "requires a running Redis instance on localhost:6379"]
    async fn factory_builds_redis_cache_skeleton() {
        let mut config = minimal_config();
        config.cache_backend = CacheBackend::Redis;
        let cache = build_exact_cache(&config);
        // Skeleton always returns None.
        assert!(cache.get("key").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn factory_builds_embedded_router() {
        let config = minimal_config();
        let client = reqwest::Client::new();
        let router = build_slm_router(&config, &client);
        let label = router.classify_intent("Hello").await.unwrap();
        assert_eq!(label, "COMPLEX");
    }

    #[tokio::test]
    async fn factory_builds_vllm_router() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": "COMPLEX" }
                }]
            })))
            .mount(&mock_server)
            .await;

        let mut config = minimal_config();
        config.router_backend = RouterBackend::Vllm;
        config.vllm_url = mock_server.uri();
        let client = reqwest::Client::new();
        let router = build_slm_router(&config, &client);
        let label = router.classify_intent("Hello").await.unwrap();
        assert_eq!(label, "COMPLEX");
    }
}

use std::sync::Arc;

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::models::FinalLayer;
use crate::state::AppState;

/// Layer 0 — Authentication middleware.
///
/// Validates the `X-API-Key` request header against the configured
/// `gateway_api_key`. If the key is missing or does not match, the
/// Deflection Stack is short-circuited with a `401 Unauthorized` response.
pub async fn auth_middleware(request: Request, next: Next) -> Response {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            tracing::error!("Layer 0: AppState missing from request extensions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "Internal Server Error",
                    "message": "Firewall misconfiguration: missing application state"
                })),
            )
                .into_response();
        }
    };

    let api_key = request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if key == state.config.gateway_api_key => {
            tracing::debug!("Layer 0: API key validated");
            next.run(request).await
        }
        _ => {
            tracing::warn!("Layer 0: Unauthorized – invalid or missing API key");
            let mut response = (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "Unauthorized",
                    "message": "Missing or invalid X-API-Key header"
                })),
            )
                .into_response();
            response.extensions_mut().insert(FinalLayer::AuthBlocked);
            response
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, middleware as axum_mw, routing::post, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::clients::slm::SlmClient;
    use crate::config::{AppConfig, CacheMode, EmbeddingSidecarSettings, Layer2Settings};
    use crate::layer1::layer1a_cache::ExactMatchCache;
    use crate::state::AppLlmAgent;
    use crate::vector_cache::VectorCache;
    use std::num::NonZeroUsize;

    use crate::layer1::embeddings::shared_test_embedder;

    /// Minimal mock LLM agent.
    struct MockAgent;

    #[async_trait::async_trait]
    impl AppLlmAgent for MockAgent {
        async fn chat(&self, _prompt: &str) -> anyhow::Result<String> {
            Ok("mock".into())
        }
        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    /// Build a minimal AppState with the given API key.
    fn test_state(api_key: &str) -> Arc<AppState> {
        let config = Arc::new(AppConfig {
            host_port: "127.0.0.1:0".into(),
            inference_engine: crate::config::InferenceEngineMode::Sidecar,
            gateway_api_key: api_key.into(),
            cache_mode: CacheMode::Exact,
            cache_backend: crate::config::CacheBackend::Memory,
            redis_url: "redis://127.0.0.1:6379".into(),
            router_backend: crate::config::RouterBackend::Embedded,
            vllm_url: "http://127.0.0.1:8000".into(),
            vllm_model: "gemma-2-2b-it".into(),
            embedding_model: "test".into(),
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
        });

        Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
            vector_cache: Arc::new(VectorCache::new(0.85, 300, 100)),
            llm_agent: Arc::new(MockAgent),
            slm_client: Arc::new(SlmClient::new(&config.layer2)),
            text_embedder: shared_test_embedder(),
            config,
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        })
    }

    /// Build a router with auth middleware and a simple OK handler.
    fn test_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/test", post(|| async { "ok" }))
            .layer(axum_mw::from_fn(auth_middleware))
            .layer(axum_mw::from_fn(move |mut req: Request, next: Next| {
                let st = state.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            }))
    }

    #[tokio::test]
    async fn valid_api_key_passes_through() {
        let state = test_state("secret-key");
        let app = test_app(state);

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .header("X-API-Key", "secret-key")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn missing_api_key_returns_401() {
        let state = test_state("secret-key");
        let app = test_app(state);

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"], "Unauthorized");
    }

    #[tokio::test]
    async fn invalid_api_key_returns_401() {
        let state = test_state("correct-key");
        let app = test_app(state);

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .header("X-API-Key", "wrong-key")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn empty_api_key_matches_empty_config() {
        // Both configured key and provided key are empty strings — should pass.
        let state = test_state("");
        let app = test_app(state);

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .header("X-API-Key", "")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

use std::sync::Arc;
use std::time::Instant;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use sha2::{Digest, Sha256};
use tracing::{info_span, Instrument};

use crate::core::retry::{execute_with_retry, RetryConfig};
use crate::errors::GatewayError;
use crate::middleware::body_buffer::BufferedBody;
use crate::models::{ChatResponse, FinalLayer};
use crate::state::AppState;

/// Layer 3 — Fallback handler.
///
/// Runs **only** if every preceding middleware layer decided it could
/// not handle the request. Dispatches the prompt to the configured
/// LLM provider via `rig-core`.
pub async fn chat_handler(request: Request) -> impl IntoResponse {
    let span = info_span!(
        "layer3_llm",
        ai.prompt.length_bytes = tracing::field::Empty,
        provider.name = tracing::field::Empty,
        model = tracing::field::Empty,
    );
    async move {
        let layer_start = Instant::now();
        let state = match request.extensions().get::<Arc<AppState>>() {
            Some(s) => s.clone(),
            None => {
                tracing::error!("Layer 3: AppState missing from request extensions");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ChatResponse {
                        layer: 3,
                        message: "Gateway misconfiguration: missing application state".into(),
                        model: None,
                    }),
                )
                    .into_response();
            }
        };

        // ------------------------------------------------------------------
    // 1. Extract the prompt from the buffered body (set by body_buffer
    //    middleware). No body-stream consumption needed.
    // ------------------------------------------------------------------
    let body_bytes = match request.extensions().get::<BufferedBody>() {
        Some(buf) => buf.0.clone(),
        None => {
            tracing::error!("Layer 3: BufferedBody missing from request extensions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ChatResponse {
                    layer: 3,
                    message: "Gateway misconfiguration: missing buffered body".into(),
                    model: None,
                }),
            )
                .into_response();
        }
    };

    // Try JSON `{ "prompt": "..." }`, fall back to raw string.
    let prompt: String = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("prompt").and_then(|p| p.as_str()).map(String::from))
        .unwrap_or_else(|| String::from_utf8_lossy(&body_bytes).to_string());

    tracing::Span::current().record("ai.prompt.length_bytes", prompt.len() as u64);

    let provider_name = state.llm_agent.provider_name();
    tracing::Span::current().record("provider.name", provider_name);
    tracing::Span::current().record("model", state.config.external_llm_model.as_str());
    tracing::info!(prompt = %prompt, provider = provider_name, "Layer 3: Forwarding to LLM via Rig");

    // ------------------------------------------------------------------
    // 2. Dispatch to the configured rig-core Agent — with retry.
    // ------------------------------------------------------------------
    let retry_cfg = RetryConfig::default();
    let agent = state.llm_agent.clone();
    let provider_for_err = provider_name.to_string();
    let prompt_for_retry = prompt.clone();

    let result = execute_with_retry(&retry_cfg, "L3_Cloud_LLM", || {
        let agent = agent.clone();
        let prompt = prompt_for_retry.clone();
        let provider = provider_for_err.clone();
        async move {
            agent
                .chat(&prompt)
                .await
                .map_err(|e| GatewayError::from_llm_error(&provider, &e))
        }
    })
    .await;

    match result {
        Ok(text) => {
            crate::metrics::record_layer_duration("L3_Cloud", layer_start.elapsed());
            let mut response = (
                StatusCode::OK,
                Json(ChatResponse {
                    layer: 3,
                    message: text,
                    model: Some(state.config.external_llm_model.clone()),
                }),
            )
                .into_response();
            response.extensions_mut().insert(FinalLayer::Cloud);
            response
        }
        Err(gw_err) => {
            crate::metrics::record_layer_duration("L3_Cloud", layer_start.elapsed());
            crate::metrics::record_error(gw_err.layer_label(), if gw_err.is_retryable() { "retryable" } else { "fatal" });
            tracing::error!(error = %gw_err, provider = provider_name, "Layer 3: LLM call failed after retries");

            // ── Stale-cache fallback ─────────────────────────────
            // If the LLM is down, try to serve a previously-cached
            // answer for this exact prompt so the user still gets
            // *something* useful.
            let exact_key = hex::encode(Sha256::digest(prompt.as_bytes()));
            if let Some(cached) = state.exact_cache.get(&exact_key) {
                tracing::info!(
                    cache.key = %exact_key,
                    "Layer 3: Serving stale cache entry as fallback"
                );
                crate::metrics::record_error("L3_StaleFallback", "fallback_used");
                let mut response = (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    cached,
                )
                    .into_response();
                response.extensions_mut().insert(FinalLayer::Cloud);
                return response;
            }

            let mut response = (
                StatusCode::BAD_GATEWAY,
                Json(ChatResponse {
                    layer: 3,
                    message: format!("[{provider_name}] {gw_err}"),
                    model: None,
                }),
            )
                .into_response();
            response.extensions_mut().insert(FinalLayer::Cloud);
            response
        }
    }
    }
    .instrument(span)
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, middleware as axum_mw, routing::post, Router};
    use http_body_util::BodyExt;
    use sha2::{Digest, Sha256};
    use tower::ServiceExt;

    use crate::clients::slm::SlmClient;
    use crate::config::{AppConfig, CacheMode, EmbeddingSidecarSettings, Layer2Settings};
    use crate::layer1::embeddings::shared_test_embedder;
    use crate::layer1::layer1a_cache::ExactMatchCache;
    use crate::middleware::body_buffer::buffer_body_middleware;
    use crate::state::AppLlmAgent;
    use crate::vector_cache::VectorCache;
    use std::num::NonZeroUsize;

    struct SuccessAgent;

    #[async_trait::async_trait]
    impl AppLlmAgent for SuccessAgent {
        async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
            Ok(format!("Reply to: {prompt}"))
        }
        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    struct FailAgent;

    #[async_trait::async_trait]
    impl AppLlmAgent for FailAgent {
        async fn chat(&self, _prompt: &str) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("provider outage"))
        }
        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    fn test_state(agent: Arc<dyn AppLlmAgent>) -> Arc<AppState> {
        let config = Arc::new(AppConfig {
            host_port: "127.0.0.1:0".into(),
            inference_engine: crate::config::InferenceEngineMode::Sidecar,
            gateway_api_key: "test".into(),
            cache_mode: CacheMode::Exact,
            cache_backend: crate::config::CacheBackend::Memory,
            redis_url: "redis://127.0.0.1:6379".into(),
            router_backend: crate::config::RouterBackend::Embedded,
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
            external_llm_model: "gpt-4o-mini".into(),
            external_llm_api_key: "".into(),
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
        });

        Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
            vector_cache: Arc::new(VectorCache::new(0.85, 300, 100)),
            llm_agent: agent,
            slm_client: Arc::new(SlmClient::new(&config.layer2)),
            text_embedder: shared_test_embedder(),
            config,
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        })
    }

    fn handler_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/api/chat", post(chat_handler))
            .layer(axum_mw::from_fn(buffer_body_middleware))
            .layer(axum_mw::from_fn(
                move |mut req: Request, next: axum_mw::Next| {
                    let st = state.clone();
                    async move {
                        req.extensions_mut().insert(st);
                        next.run(req).await
                    }
                },
            ))
    }

    #[tokio::test]
    async fn successful_llm_response() {
        let state = test_state(Arc::new(SuccessAgent));
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": "hello" })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["layer"], 3);
        assert_eq!(json["message"], "Reply to: hello");
        assert_eq!(json["model"], "gpt-4o-mini");
    }

    #[tokio::test]
    async fn llm_failure_returns_502() {
        let state = test_state(Arc::new(FailAgent));
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": "test" })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["layer"], 3);
        assert!(json["message"]
            .as_str()
            .unwrap()
            .contains("provider outage"));
    }

    #[tokio::test]
    async fn raw_string_body_used_as_prompt() {
        let state = test_state(Arc::new(SuccessAgent));
        let app = handler_app(state);

        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .body(Body::from("raw text prompt"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["message"], "Reply to: raw text prompt");
    }

    #[tokio::test]
    async fn stale_cache_fallback_on_llm_failure() {
        let state = test_state(Arc::new(FailAgent));

        // Pre-populate the exact cache with a stale entry for "fallback test".
        let prompt = "fallback test";
        let key = hex::encode(Sha256::digest(prompt.as_bytes()));
        let cached_json = serde_json::to_string(&ChatResponse {
            layer: 3,
            message: "stale cached answer".into(),
            model: Some("gpt-4o-mini".into()),
        })
        .unwrap();
        state.exact_cache.put(key, cached_json);

        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": prompt })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should get 200 (stale cache) instead of 502.
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&body_bytes);
        assert!(text.contains("stale cached answer"));
    }

    #[tokio::test]
    async fn no_stale_cache_returns_502() {
        // When the LLM fails and there is no stale cache entry, 502 is expected.
        let state = test_state(Arc::new(FailAgent));
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": "no-cache-entry" })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }
}

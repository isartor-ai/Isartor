use std::sync::Arc;
use std::time::Instant;

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::IntoResponse;
use tracing::Instrument;

use crate::metrics;
use crate::models::FinalLayer;
use crate::state::AppState;

/// Root monitoring middleware — **outermost** layer in the Axum stack.
///
/// Responsibilities:
///   1. Open a parent trace span (`gateway_request`) that wraps the
///      entire request lifetime, carrying standard HTTP attributes
///      **and** the custom `isartor.final_layer` tag.
///   2. After the response returns, read the [`FinalLayer`] extension
///      that child middlewares (cache / SLM / handler) inserted to
///      determine *which* firewall layer handled the request.
///   3. Record OTel metrics via the global `GatewayMetrics` singleton:
///      - `isartor_requests_total`
///      - `isartor_request_duration_seconds`
///      - `isartor_tokens_saved_total` (when resolved before L3)
pub async fn root_monitoring_middleware(request: Request, next: Next) -> impl IntoResponse {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let client_addr = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    // Extract the prompt length for tokens-saved estimation.
    // We peek at the Content-Length header as a rough proxy (the body
    // hasn't been consumed yet).
    let content_length: u64 = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Create the root span with all required HTTP + business attributes.
    // `isartor.final_layer` and `http.status_code` are recorded after
    // the response returns.
    let root_span = tracing::info_span!(
        "gateway_request",
        http.method = %method,
        http.route = %path,
        http.status_code = tracing::field::Empty,
        client.address = %client_addr,
        isartor.final_layer = tracing::field::Empty,
    );

    let state_opt = request.extensions().get::<Arc<AppState>>().cloned();

    let start = Instant::now();
    // Run the entire middleware + handler chain inside the root span.
    // Use `.instrument()` so the span is active across the await without
    // holding a non-Send guard.
    let response = next.run(request).instrument(root_span.clone()).await;
    let elapsed = start.elapsed();

    // ── Determine final layer ────────────────────────────────────
    let final_layer = response
        .extensions()
        .get::<FinalLayer>()
        .copied()
        .unwrap_or_else(|| {
            if response.status().is_client_error() {
                FinalLayer::AuthBlocked
            } else {
                FinalLayer::Cloud
            }
        });

    let layer_label = final_layer.as_str();
    let status_code = response.status().as_u16();

    // ── Record span attributes ───────────────────────────────────
    root_span.record("http.status_code", status_code);
    root_span.record("isartor.final_layer", layer_label);

    // ── Record OTel metrics ──────────────────────────────────────
    metrics::record_request(layer_label, status_code, elapsed.as_secs_f64());

    // Record tokens saved when request was resolved before Layer 3.
    let resolved_early = matches!(
        final_layer,
        FinalLayer::ExactCache | FinalLayer::SemanticCache | FinalLayer::Slm
    );
    if resolved_early {
        // Estimate using prompt size or a conservative default.
        let estimated_tokens = if content_length > 0 {
            metrics::estimate_tokens(
                &"x".repeat(content_length as usize), // rough char-count proxy
            )
        } else {
            256 // conservative default
        };
        metrics::record_tokens_saved(layer_label, estimated_tokens);
    }

    tracing::info!(
        http.method = %method,
        http.route = %path,
        http.status_code = status_code,
        isartor.final_layer = layer_label,
        duration_ms = elapsed.as_millis() as u64,
        monitoring = state_opt.as_ref().is_some_and(|s| s.config.enable_monitoring),
        "Request completed"
    );

    // ── Attach observability headers ─────────────────────────────
    // X-Isartor-Layer: which layer resolved the request (l1a/l1b/l2/l3)
    // X-Isartor-Deflected: true when the request was resolved without
    //                      reaching the cloud LLM
    let layer_header_value = match final_layer {
        FinalLayer::ExactCache => "l1a",
        FinalLayer::SemanticCache => "l1b",
        FinalLayer::Slm => "l2",
        FinalLayer::Cloud | FinalLayer::AuthBlocked => "l3",
    };
    let deflected_header_value = if resolved_early { "true" } else { "false" };

    let mut response = response;
    response.headers_mut().insert(
        "X-Isartor-Layer",
        HeaderValue::from_static(layer_header_value),
    );
    response.headers_mut().insert(
        "X-Isartor-Deflected",
        HeaderValue::from_static(deflected_header_value),
    );

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::{body::Body, middleware as axum_mw, routing::post, Json, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::clients::slm::SlmClient;
    use crate::config::{AppConfig, CacheMode, EmbeddingSidecarSettings, Layer2Settings};
    use crate::layer1::embeddings::shared_test_embedder;
    use crate::layer1::layer1a_cache::ExactMatchCache;
    use crate::models::ChatResponse;
    use crate::state::AppLlmAgent;
    use crate::vector_cache::VectorCache;
    use std::num::NonZeroUsize;

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

    fn test_state(enable_monitoring: bool) -> Arc<AppState> {
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
            external_llm_model: "test".into(),
            external_llm_api_key: "".into(),
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring,
            enable_slm_router: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
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

    fn monitoring_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route(
                "/api/chat",
                post(|| async {
                    (
                        StatusCode::OK,
                        Json(ChatResponse {
                            layer: 3,
                            message: "ok".into(),
                            model: None,
                        }),
                    )
                }),
            )
            .layer(axum_mw::from_fn(root_monitoring_middleware))
            .layer(axum_mw::from_fn(
                move |mut req: axum::extract::Request, next: axum_mw::Next| {
                    let st = state.clone();
                    async move {
                        req.extensions_mut().insert(st);
                        next.run(req).await
                    }
                },
            ))
    }

    #[tokio::test]
    async fn monitoring_passes_through_disabled() {
        let state = test_state(false);
        let app = monitoring_app(state);

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["layer"], 3);
    }

    #[tokio::test]
    async fn monitoring_passes_through_enabled() {
        let state = test_state(true);
        let app = monitoring_app(state);

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn monitoring_records_client_error_as_auth_blocked() {
        let state = test_state(false);
        // Build an app that returns 401
        let app = Router::new()
            .route(
                "/api/chat",
                post(|| async { StatusCode::UNAUTHORIZED.into_response() }),
            )
            .layer(axum_mw::from_fn(root_monitoring_middleware))
            .layer(axum_mw::from_fn(
                move |mut req: axum::extract::Request, next: axum_mw::Next| {
                    let st = state.clone();
                    async move {
                        req.extensions_mut().insert(st);
                        next.run(req).await
                    }
                },
            ));

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}

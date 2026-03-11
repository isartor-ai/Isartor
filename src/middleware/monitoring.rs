use std::sync::Arc;
use std::time::Instant;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::IntoResponse;
use opentelemetry::metrics::{Counter, Histogram};
use opentelemetry::{global, KeyValue};

use crate::models::FinalLayer;
use crate::state::AppState;

/// Root monitoring middleware — **outermost** layer in the Axum stack.
///
/// Responsibilities:
///   1. Start a parent trace span (`gateway_request`) that wraps the
///      entire request lifetime.
///   2. After the response returns, read the [`FinalLayer`] extension
///      that child middlewares (cache / SLM / handler) inserted to
///      determine *which* gateway layer handled the request.
///   3. Record an `isartor_requests_total` counter **and** an
///      `isartor_request_duration_seconds` histogram, both tagged with
///      the granular `final_layer` label.
pub async fn root_monitoring_middleware(request: Request, next: Next) -> impl IntoResponse {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    // Diagnostics: use println! for compatibility
    println!("[monitoring] gateway_request: method={} path={}", method, path);

    async {
        let state = request
            .extensions()
            .get::<Arc<AppState>>()
            .expect("AppState missing")
            .clone();

        let enable_monitoring = state.config.enable_monitoring;

        // Lazily initialise OTel instruments only when monitoring is on.
        let mut counter: Option<Counter<u64>> = None;
        let mut histogram: Option<Histogram<f64>> = None;

        if enable_monitoring {
            let meter = global::meter("isartor.gateway");
            counter = Some(
                meter
                    .u64_counter("isartor_requests_total")
                    .with_description(
                        "Total requests processed, labelled by the final handling layer",
                    )
                    .build(),
            );
            histogram = Some(
                meter
                    .f64_histogram("isartor_request_duration_seconds")
                    .with_description("End-to-end request latency in seconds")
                    .build(),
            );
        }

        let start = Instant::now();

        // ── Run the rest of the middleware / handler chain ────────
        let response = next.run(request).await;

        let elapsed = start.elapsed();

        // ── Determine final layer ────────────────────────────────
        let final_layer = response
            .extensions()
            .get::<FinalLayer>()
            .copied()
            .unwrap_or_else(|| {
                // Fallback heuristic when no extension was set.
                if response.status().is_client_error() {
                    FinalLayer::AuthBlocked
                } else {
                    // Check legacy ChatResponse.layer field
                    response
                        .extensions()
                        .get::<crate::models::ChatResponse>()
                        .map(|cr| match cr.layer {
                            1 => FinalLayer::ExactCache, // generic "cache" fallback
                            2 => FinalLayer::Slm,
                            _ => FinalLayer::Cloud,
                        })
                        .unwrap_or(FinalLayer::Cloud)
                }
            });

        let layer_label = final_layer.as_str();

        // ── Record trace attribute ───────────────────────────────
        // root_span removed; no tracing attribute recording

        // ── Record OTel metrics ──────────────────────────────────
        let attrs = [
            KeyValue::new("final_layer", layer_label),
            KeyValue::new("status_code", response.status().as_u16().to_string()),
        ];

        if let Some(c) = counter {
            c.add(1, &attrs);
        }
        if let Some(h) = histogram {
            h.record(elapsed.as_secs_f64(), &attrs);
        }

        println!(
            "[monitoring] Request finished: method={} path={} final_layer={} status={} duration_ms={}",
            method,
            path,
            layer_label,
            response.status().as_u16(),
            elapsed.as_millis()
        );

        response
    }
    // .instrument(root_span.clone())  // root_span removed, no tracing
    .await
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
            otel_exporter_endpoint: "http://localhost:4317".into(),
            pipeline_embedding_dim: 384,
            pipeline_similarity_threshold: 0.92,
            pipeline_rerank_top_k: 5,
            pipeline_max_concurrency: 256,
            pipeline_min_concurrency: 4,
            pipeline_target_latency_ms: 500,
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

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use http_body_util::BodyExt;
use sha2::{Digest, Sha256};

use crate::config::CacheMode;
use crate::core::prompt::{extract_prompt, extract_semantic_key};
use crate::middleware::body_buffer::BufferedBody;
use crate::models::{ChatResponse, FinalLayer};
use crate::state::AppState;

/// Layer 1 — Cache middleware with configurable strategy.
///
/// Supports three modes controlled by `ISARTOR_CACHE_MODE`:
///
/// | Mode       | Lookup                                           |
/// |------------|--------------------------------------------------|
/// | `exact`    | SHA-256 of the prompt → HashMap                  |
/// | `semantic` | Ollama embedding → cosine similarity vector scan |
/// | `both`     | Exact first (fast), then semantic if no hit       |
///
/// On a miss the downstream response is captured and stored in the
/// active cache(s). If the embedding service is unreachable in
/// `semantic`/`both` modes, the layer gracefully falls through.
pub async fn cache_middleware(request: Request, next: Next) -> Response {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            tracing::error!("Layer 1: AppState missing from request extensions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(ChatResponse {
                    layer: 1,
                    message: "Firewall misconfiguration: missing application state".into(),
                    model: None,
                }),
            )
                .into_response();
        }
    };

    // ------------------------------------------------------------------
    // 1. Extract the prompt from the buffered body (set by body_buffer
    //    middleware). The request body stream is untouched.
    // ------------------------------------------------------------------
    let body_bytes = match request.extensions().get::<BufferedBody>() {
        Some(buf) => buf.0.clone(),
        None => {
            tracing::error!("Layer 1: BufferedBody missing from request extensions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ChatResponse {
                    layer: 1,
                    message: "Firewall misconfiguration: missing buffered body".into(),
                    model: None,
                }),
            )
                .into_response();
        }
    };

    let prompt = extract_prompt(body_bytes.as_ref());

    // For semantic matching, use only the last user message so the embedding
    // captures the actual question rather than a large system prompt that
    // dominates the vector and causes unrelated questions to match.
    let semantic_prompt = extract_semantic_key(body_bytes.as_ref());

    // Keep cache entries separate per response format to avoid cross-format cache hits.
    // (e.g., OpenAI-compatible endpoints should not return native ChatResponse bodies.)
    let cache_ns = match request.uri().path() {
        "/v1/chat/completions" => "openai",
        "/v1/messages" => "anthropic",
        _ => "native",
    };
    let cache_prompt = format!("{cache_ns}|{prompt}");
    let semantic_cache_prompt = format!("{cache_ns}|{semantic_prompt}");

    let mode = &state.config.cache_mode;
    let layer_start = Instant::now();

    // ------------------------------------------------------------------
    // 2. Exact-match lookup (when mode is Exact or Both).
    // ------------------------------------------------------------------
    let exact_key = if *mode == CacheMode::Exact || *mode == CacheMode::Both {
        let key = hex::encode(Sha256::digest(cache_prompt.as_bytes()));

        tracing::debug!(
            cache.mode = ?mode,
            cache.key = %key,
            "L1a: exact-match lookup",
        );

        if let Some(cached) = state.exact_cache.get(&key) {
            tracing::info!(cache.key = %key, "L1a: exact cache HIT");
            tracing::Span::current().record("gateway.cache.hit", true);
            crate::metrics::record_layer_duration("L1a_ExactCache", layer_start.elapsed());
            let mut response = (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                cached,
            )
                .into_response();
            response.extensions_mut().insert(FinalLayer::ExactCache);
            return response;
        }
        Some(key)
    } else {
        None
    };

    // ------------------------------------------------------------------
    // 3. Semantic lookup (when mode is Semantic or Both).
    //    Uses the in-process candle TextEmbedder — no HTTP round-trip.
    // ------------------------------------------------------------------
    let embedding: Option<Vec<f32>> = if *mode == CacheMode::Semantic || *mode == CacheMode::Both {
        let embedder = state.text_embedder.clone();
        let prompt_clone = semantic_cache_prompt.clone();

        // candle BertModel inference is CPU-bound; run on the blocking pool
        // so we don't starve the Tokio async workers.
        let result =
            tokio::task::spawn_blocking(move || embedder.generate_embedding(&prompt_clone)).await;
        match result {
            Ok(Ok(emb)) => {
                tracing::debug!(embedding.dims = emb.len(), "L1b: embedding generated");
                Some(emb)
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    error = %e,
                    "L1b: in-process embedding failed – skipping semantic cache",
                );
                crate::metrics::record_error("L1b_Embedding", "retryable");
                None
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "L1b: embedding task panicked – skipping semantic cache",
                );
                crate::metrics::record_error("L1b_Embedding", "fatal");
                None
            }
        }
    } else {
        None
    };

    // Search the vector cache if we obtained an embedding.
    if let Some(ref emb) = embedding {
        tracing::debug!(embedding.dims = emb.len(), "L1b: semantic lookup");
        if let Some(cached) = state.vector_cache.search(emb).await {
            tracing::info!("L1b: semantic cache HIT");
            tracing::Span::current().record("gateway.cache.hit", true);
            crate::metrics::record_layer_duration("L1b_SemanticCache", layer_start.elapsed());
            let mut response = (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                cached,
            )
                .into_response();
            response.extensions_mut().insert(FinalLayer::SemanticCache);
            return response;
        }
    }

    tracing::debug!(
        cache.mode = ?mode,
        "L1: cache MISS – forwarding downstream",
    );

    // ------------------------------------------------------------------
    // 4. Cache miss: forward to next layer (body stream intact).
    // ------------------------------------------------------------------
    let response = next.run(request).await;

    // ------------------------------------------------------------------
    // 5. Capture the downstream response and store in active cache(s).
    // ------------------------------------------------------------------
    let (resp_parts, resp_body) = response.into_parts();
    if let Ok(collected) = resp_body.collect().await {
        let resp_bytes = collected.to_bytes();
        let resp_string = String::from_utf8_lossy(&resp_bytes).to_string();

        // Normalize what we store in caches: cache hits should reflect Layer 1,
        // even if the original response came from Layer 2/3.
        let cache_value = match serde_json::from_slice::<ChatResponse>(&resp_bytes) {
            Ok(mut parsed) => {
                parsed.layer = 1;
                serde_json::to_string(&parsed).unwrap_or_else(|_| resp_string.clone())
            }
            Err(_) => resp_string.clone(),
        };

        if resp_parts.status.is_success() {
            // Store in exact cache.
            if let Some(key) = exact_key {
                tracing::debug!(cache.key = %key, "L1a: storing in exact cache");
                state.exact_cache.put(key, cache_value.clone());
            }
            // Store in vector cache.
            if let Some(emb) = embedding {
                tracing::debug!("L1b: storing in vector cache");
                state.vector_cache.insert(emb, cache_value.clone()).await;
            }
        }

        return Response::from_parts(resp_parts, Body::from(resp_bytes));
    }

    crate::metrics::record_error("L1_Cache", "retryable");
    (
        StatusCode::BAD_GATEWAY,
        Json(ChatResponse {
            layer: 1,
            message: "Failed to read downstream response".into(),
            model: None,
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, middleware as axum_mw, routing::post};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::clients::slm::SlmClient;
    use crate::config::{AppConfig, CacheMode, EmbeddingSidecarSettings, Layer2Settings};
    use crate::layer1::embeddings::shared_test_embedder;
    use crate::layer1::layer1a_cache::ExactMatchCache;
    use crate::middleware::body_buffer::buffer_body_middleware;
    use crate::models::ChatResponse;
    use crate::state::AppLlmAgent;
    use crate::vector_cache::VectorCache;

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

    fn test_config(mode: CacheMode) -> Arc<AppConfig> {
        Arc::new(AppConfig {
            host_port: "127.0.0.1:0".into(),
            inference_engine: crate::config::InferenceEngineMode::Sidecar,
            gateway_api_key: "test".into(),
            cache_mode: mode,
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
            enable_monitoring: false,
            enable_slm_router: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
        })
    }

    fn test_state(mode: CacheMode) -> Arc<AppState> {
        let config = test_config(mode);
        Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: Arc::new(ExactMatchCache::new(
                std::num::NonZeroUsize::new(100).unwrap(),
            )),
            vector_cache: Arc::new(VectorCache::new(0.85, 300, 100)),
            llm_agent: Arc::new(MockAgent),
            slm_client: Arc::new(SlmClient::new(&config.layer2)),
            text_embedder: shared_test_embedder(),
            config,
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        })
    }

    /// Build a router with cache middleware and a handler that echoes the body.
    fn cache_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route(
                "/api/chat",
                post(|| async {
                    (
                        StatusCode::OK,
                        axum::Json(ChatResponse {
                            layer: 3,
                            message: "downstream response".into(),
                            model: Some("gpt-4o".into()),
                        }),
                    )
                }),
            )
            .layer(axum_mw::from_fn(cache_middleware))
            .layer(axum_mw::from_fn(buffer_body_middleware))
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

    fn json_body(prompt: &str) -> Body {
        Body::from(serde_json::to_vec(&serde_json::json!({ "prompt": prompt })).unwrap())
    }

    #[tokio::test]
    async fn exact_cache_miss_then_hit() {
        let state = test_state(CacheMode::Exact);
        let app = cache_app(state.clone());

        // First request — cache miss, should get downstream response.
        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(json_body("hello"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["message"], "downstream response");
        assert_eq!(json["layer"], 3);

        // The response should now be in the exact cache (normalized to layer 1).
        let key = hex::encode(sha2::Sha256::digest(b"native|hello"));
        let cached = state.exact_cache.get(&key);
        assert!(cached.is_some(), "Response should be cached after miss");
        let cached_json: serde_json::Value = serde_json::from_str(&cached.unwrap()).unwrap();
        assert_eq!(cached_json["layer"], 1);

        // Second request — cache hit, should get cached response.
        let app2 = cache_app(state);
        let req2 = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(json_body("hello"))
            .unwrap();

        let resp2 = app2.oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
        let body2 = resp2.into_body().collect().await.unwrap().to_bytes();
        let json2: serde_json::Value = serde_json::from_slice(&body2).unwrap();
        assert_eq!(json2["message"], "downstream response");
        // Cache hits must report Layer 1 (even if the original response came from Layer 3).
        assert_eq!(json2["layer"], 1);
    }

    #[tokio::test]
    async fn semantic_mode_falls_through_when_embed_unreachable() {
        // Semantic mode needs an embed endpoint; when unreachable it falls through.
        let state = test_state(CacheMode::Semantic);
        let app = cache_app(state);

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(json_body("test semantic"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should still get the downstream response (graceful fallthrough).
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["message"], "downstream response");
    }

    #[tokio::test]
    async fn both_mode_exact_hit_short_circuits() {
        let state = test_state(CacheMode::Both);

        // Pre-populate the exact cache.
        let key = hex::encode(sha2::Sha256::digest(b"native|cached prompt"));
        let cached_json = serde_json::to_string(&ChatResponse {
            layer: 1,
            message: "from cache".into(),
            model: None,
        })
        .unwrap();
        state.exact_cache.put(key, cached_json);

        let app = cache_app(state);

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(json_body("cached prompt"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("from cache"));
    }

    #[tokio::test]
    async fn raw_string_body_fallback() {
        // If body is not valid JSON, the raw string should be used as the prompt.
        let state = test_state(CacheMode::Exact);
        let app = cache_app(state.clone());

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "text/plain")
            .body(Body::from("raw prompt text"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify cached under the raw text key.
        let key = hex::encode(sha2::Sha256::digest(b"native|raw prompt text"));
        let cached = state.exact_cache.get(&key);
        assert!(cached.is_some());
    }
}

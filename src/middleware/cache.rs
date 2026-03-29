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

use crate::anthropic_sse;
use crate::config::CacheMode;
use crate::core::cache_scope::{
    build_exact_cache_key, extract_session_cache_scope, namespaced_semantic_cache_input,
};
use crate::core::prompt::{
    extract_cache_key, extract_request_model, extract_semantic_key, has_tooling,
    override_request_model,
};
use crate::middleware::body_buffer::BufferedBody;
use crate::models::{ChatResponse, FinalLayer};
use crate::openai_sse;
use crate::state::AppState;

fn streaming_cache_response(
    cache_ns: &str,
    cached_json: &str,
    model_fallback: &str,
) -> Option<Response> {
    match cache_ns {
        "anthropic" => Some(anthropic_sse::cached_to_sse_response(
            cached_json,
            model_fallback,
        )),
        "openai" => Some(openai_sse::cached_to_sse_response(
            cached_json,
            model_fallback,
        )),
        _ => None,
    }
}

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
    let user_agent = request
        .headers()
        .get(axum::http::header::USER_AGENT)
        .and_then(|value| value.to_str().ok());
    let tool = crate::tool_identity::identify_tool_or_fallback(user_agent, "gateway");
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

    let resolved_model = extract_request_model(body_bytes.as_ref())
        .map(|model| state.config.resolve_model_alias(&model))
        .unwrap_or_else(|| state.config.configured_model_id());
    let canonical_body = override_request_model(body_bytes.as_ref(), &resolved_model);
    let cache_key_material = extract_cache_key(&canonical_body);
    let has_tooling = has_tooling(&canonical_body);
    let session_cache_scope = extract_session_cache_scope(request.headers(), &canonical_body);

    // For semantic matching, use only the last user message so the embedding
    // captures the actual question rather than a large system prompt that
    // dominates the vector and causes unrelated questions to match.
    let semantic_prompt = format!(
        "model: {resolved_model}\n{}",
        extract_semantic_key(&canonical_body)
    );

    // Keep cache entries separate per response format to avoid cross-format cache hits.
    // (e.g., OpenAI-compatible endpoints should not return native ChatResponse bodies.)
    let cache_ns = match request.uri().path() {
        "/v1/chat/completions" => "openai",
        "/v1/messages" => "anthropic",
        _ => "native",
    };
    let semantic_cache_prompt = namespaced_semantic_cache_input(cache_ns, &semantic_prompt);
    let semantic_cache_enabled = request.uri().path() != "/v1/messages" && !has_tooling;

    // Detect if the client expects an SSE stream (Claude Code sends stream: true).
    let is_streaming = match cache_ns {
        "anthropic" => anthropic_sse::is_streaming_request(&canonical_body),
        "openai" => openai_sse::is_streaming_request(&canonical_body),
        _ => false,
    };

    let mode = &state.config.cache_mode;
    let layer_start = Instant::now();

    // ------------------------------------------------------------------
    // 2. Exact-match lookup (when mode is Exact or Both).
    // ------------------------------------------------------------------
    let exact_key = if *mode == CacheMode::Exact || *mode == CacheMode::Both {
        let key = build_exact_cache_key(
            cache_ns,
            &cache_key_material,
            session_cache_scope.as_deref(),
        );

        tracing::debug!(
            cache.mode = ?mode,
            cache.key = %key,
            cache.session_scoped = session_cache_scope.is_some(),
            "L1a: exact-match lookup",
        );

        if let Some(cached) = state.exact_cache.get(&key) {
            tracing::info!(cache.key = %key, "L1a: exact cache HIT");
            tracing::Span::current().record("gateway.cache.hit", true);
            crate::metrics::record_layer_duration_with_tool(
                "L1a_ExactCache",
                layer_start.elapsed(),
                tool,
            );
            crate::metrics::record_cache_event_with_tool("l1a", "hit", tool);
            crate::metrics::record_cache_event_with_tool("l1", "hit", tool);
            crate::visibility::record_agent_cache_event(tool, "l1a", "hit");
            crate::visibility::record_agent_cache_event(tool, "l1", "hit");
            let mut response = if is_streaming {
                streaming_cache_response(cache_ns, &cached, &resolved_model).unwrap_or_else(|| {
                    (
                        StatusCode::OK,
                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                        cached.clone(),
                    )
                        .into_response()
                })
            } else {
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    cached,
                )
                    .into_response()
            };
            response.extensions_mut().insert(FinalLayer::ExactCache);
            return response;
        }
        crate::metrics::record_cache_event_with_tool("l1a", "miss", tool);
        crate::visibility::record_agent_cache_event(tool, "l1a", "miss");
        Some(key)
    } else {
        None
    };

    // ------------------------------------------------------------------
    // 3. Semantic lookup (when mode is Semantic or Both).
    //    Uses the in-process candle TextEmbedder — no HTTP round-trip.
    // ------------------------------------------------------------------
    let embedding: Option<Vec<f32>> = if semantic_cache_enabled
        && (*mode == CacheMode::Semantic || *mode == CacheMode::Both)
    {
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
                crate::metrics::record_error_with_tool("L1b_Embedding", "retryable", tool);
                crate::visibility::record_agent_error(tool);
                None
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "L1b: embedding task panicked – skipping semantic cache",
                );
                crate::metrics::record_error_with_tool("L1b_Embedding", "fatal", tool);
                crate::visibility::record_agent_error(tool);
                None
            }
        }
    } else {
        None
    };

    // Search the vector cache if we obtained an embedding.
    if let Some(ref emb) = embedding {
        tracing::debug!(embedding.dims = emb.len(), "L1b: semantic lookup");
        if let Some(cached) = state
            .vector_cache
            .search(emb, session_cache_scope.as_deref())
            .await
        {
            tracing::info!("L1b: semantic cache HIT");
            tracing::Span::current().record("gateway.cache.hit", true);
            crate::metrics::record_layer_duration_with_tool(
                "L1b_SemanticCache",
                layer_start.elapsed(),
                tool,
            );
            crate::metrics::record_cache_event_with_tool("l1b", "hit", tool);
            crate::metrics::record_cache_event_with_tool("l1", "hit", tool);
            crate::visibility::record_agent_cache_event(tool, "l1b", "hit");
            crate::visibility::record_agent_cache_event(tool, "l1", "hit");
            let mut response = if is_streaming {
                streaming_cache_response(cache_ns, &cached, &resolved_model).unwrap_or_else(|| {
                    (
                        StatusCode::OK,
                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                        cached.clone(),
                    )
                        .into_response()
                })
            } else {
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    cached,
                )
                    .into_response()
            };
            response.extensions_mut().insert(FinalLayer::SemanticCache);
            return response;
        }
        crate::metrics::record_cache_event_with_tool("l1b", "miss", tool);
        crate::visibility::record_agent_cache_event(tool, "l1b", "miss");
    }

    tracing::debug!(
        cache.mode = ?mode,
        "L1: cache MISS – forwarding downstream",
    );
    crate::metrics::record_cache_event_with_tool("l1", "miss", tool);
    crate::visibility::record_agent_cache_event(tool, "l1", "miss");

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
                state
                    .vector_cache
                    .insert(emb, cache_value.clone(), session_cache_scope.clone())
                    .await;
            }
        }

        // Convert JSON → SSE for streaming clients at the boundary.
        // Handlers always return JSON; conversion happens here so caches keep
        // canonical JSON regardless of the client's streaming preference.
        if is_streaming
            && resp_parts.status.is_success()
            && let Some(mut sse_resp) =
                streaming_cache_response(cache_ns, &resp_string, &resolved_model)
        {
            // Preserve extensions (FinalLayer, etc.) from the downstream response.
            *sse_resp.extensions_mut() = resp_parts.extensions;
            return sse_resp;
        }

        return Response::from_parts(resp_parts, Body::from(resp_bytes));
    }

    crate::metrics::record_error_with_tool("L1_Cache", "retryable", tool);
    crate::visibility::record_agent_error(tool);
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
    use sha2::Digest;
    use tower::ServiceExt;

    use crate::clients::slm::SlmClient;
    use crate::config::{AppConfig, CacheMode, EmbeddingSidecarSettings, Layer2Settings};
    use crate::core::cache_scope::{build_exact_cache_key, derive_session_cache_scope};
    use crate::core::context_compress::InstructionCache;
    use crate::core::prompt::{extract_cache_key, extract_semantic_key};
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
                classifier_mode: crate::config::ClassifierMode::Tiered,
                max_answer_tokens: 2048,
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
            model_aliases: std::collections::HashMap::new(),
            external_llm_api_key: "".into(),
            l3_timeout_secs: 120,
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
            enable_slm_router: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            enable_request_logs: false,
            request_log_path: "~/.isartor/request_logs".into(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
            enable_context_optimizer: true,
            context_optimizer_dedup: true,
            context_optimizer_minify: true,
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
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
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
            .route(
                "/v1/chat/completions",
                post(|| async {
                    (
                        StatusCode::OK,
                        axum::Json(serde_json::json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "downstream openai response"
                                },
                                "index": 0,
                                "finish_reason": "stop"
                            }],
                            "model": "test-model"
                        })),
                    )
                }),
            )
            .route(
                "/v1/messages",
                post(|| async {
                    (
                        StatusCode::OK,
                        axum::Json(serde_json::json!({
                            "type": "message",
                            "role": "assistant",
                            "model": "test-model",
                            "content": [{"type": "text", "text": "downstream anthropic response"}],
                            "stop_reason": "end_turn"
                        })),
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

    fn anthropic_body(prompt: &str) -> Body {
        Body::from(
            serde_json::to_vec(&serde_json::json!({
                "system": "You are Claude Code. Help with software engineering tasks.",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
            }))
            .unwrap(),
        )
    }

    fn openai_body(prompt: &str, stream: bool) -> Body {
        Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt-4o-mini",
                "stream": stream,
                "messages": [{"role": "user", "content": prompt}]
            }))
            .unwrap(),
        )
    }

    fn openai_tool_body(prompt: &str, tool_name: &str) -> Body {
        Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_prev",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"}
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_prev",
                        "content": "{\"ok\":true}"
                    },
                    {"role": "user", "content": prompt}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "parameters": {"type": "object"}
                    }
                }],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": tool_name}
                }
            }))
            .unwrap(),
        )
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
        let key = hex::encode(sha2::Sha256::digest(b"native|hello\nmodel: test"));
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
        let key = hex::encode(sha2::Sha256::digest(b"native|cached prompt\nmodel: test"));
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

    #[tokio::test]
    async fn anthropic_messages_skip_semantic_cache() {
        let state = test_state(CacheMode::Semantic);
        let app = cache_app(state);

        let req1 = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(anthropic_body("what is rust"))
            .unwrap();
        let resp1 = app.clone().oneshot(req1).await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);

        let req2 = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(anthropic_body("what is rust"))
            .unwrap();
        let resp2 = app.oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);

        let body2 = resp2.into_body().collect().await.unwrap().to_bytes();
        let json2: serde_json::Value = serde_json::from_slice(&body2).unwrap();
        assert_eq!(json2["model"], "test-model");
        assert_eq!(json2["type"], "message");
    }

    #[tokio::test]
    async fn openai_streaming_miss_returns_sse_and_caches_json() {
        let state = test_state(CacheMode::Exact);
        let app = cache_app(state.clone());

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(openai_body("hello stream", true))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"object\":\"chat.completion.chunk\""));
        assert!(text.contains("\"delta\":{\"content\":\"downstream openai response\"}"));
        assert!(text.contains("data: [DONE]"));

        let key = hex::encode(sha2::Sha256::digest(
            b"openai|user: hello stream\nmodel: gpt-4o-mini",
        ));
        let cached = state
            .exact_cache
            .get(&key)
            .expect("response should be cached");
        let cached_json: serde_json::Value = serde_json::from_str(&cached).unwrap();
        assert_eq!(cached_json["model"], "test-model");
        assert_eq!(
            cached_json["choices"][0]["message"]["content"],
            "downstream openai response"
        );
    }

    #[tokio::test]
    async fn openai_streaming_exact_cache_hit_returns_sse() {
        let state = test_state(CacheMode::Exact);
        let key = hex::encode(sha2::Sha256::digest(
            b"openai|user: cached prompt\nmodel: gpt-4o-mini",
        ));
        state.exact_cache.put(
            key,
            serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "cached openai answer"
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "model": "cached-model"
            })
            .to_string(),
        );

        let app = cache_app(state);
        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(openai_body("cached prompt", true))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"model\":\"cached-model\""));
        assert!(text.contains("\"delta\":{\"content\":\"cached openai answer\"}"));
        assert!(text.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn openai_tool_requests_use_distinct_exact_cache_keys() {
        let state = test_state(CacheMode::Exact);
        let app = cache_app(state.clone());

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(openai_tool_body("check weather", "lookup_weather"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let request_body = serde_json::to_vec(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_prev",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": "{}"}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_prev",
                    "content": "{\"ok\":true}"
                },
                {"role": "user", "content": "check weather"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup_weather"}
            }
        }))
        .unwrap();
        let miss_key = hex::encode(sha2::Sha256::digest(
            format!("openai|{}", extract_cache_key(&request_body)).as_bytes(),
        ));
        assert!(
            state.exact_cache.get(&miss_key).is_some(),
            "tool-enabled request should be cached under tooling-aware key"
        );

        let plain_key = hex::encode(sha2::Sha256::digest(b"openai|user: check weather"));
        assert!(
            state.exact_cache.get(&plain_key).is_none(),
            "tool-enabled request must not reuse plain completion key"
        );
    }

    #[tokio::test]
    async fn semantic_cache_skips_tool_enabled_openai_requests() {
        let state = test_state(CacheMode::Semantic);
        let app = cache_app(state.clone());
        let request_body = serde_json::to_vec(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_prev",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": "{}"}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_prev",
                    "content": "{\"ok\":true}"
                },
                {"role": "user", "content": "weather details"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {"type": "object"}
                }
            }]
        }))
        .unwrap();

        let req1 = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(request_body.clone()))
            .unwrap();
        let resp1 = app.clone().oneshot(req1).await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);

        let semantic_prompt = format!("openai|{}", extract_semantic_key(&request_body));
        let embedding = state
            .text_embedder
            .generate_embedding(&semantic_prompt)
            .expect("embedding should be generated");
        assert!(
            state.vector_cache.search(&embedding, None).await.is_none(),
            "tool-enabled request should not populate semantic cache"
        );
    }

    #[tokio::test]
    async fn exact_cache_isolated_by_session_scope() {
        let state = test_state(CacheMode::Exact);
        let app = cache_app(state.clone());

        let req1 = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .header("x-thread-id", "thread-a")
            .body(json_body("hello"))
            .unwrap();
        let resp1 = app.clone().oneshot(req1).await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);
        let body1 = resp1.into_body().collect().await.unwrap().to_bytes();
        let json1: serde_json::Value = serde_json::from_slice(&body1).unwrap();
        assert_eq!(json1["layer"], 3);

        let req2 = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .header("x-thread-id", "thread-a")
            .body(json_body("hello"))
            .unwrap();
        let resp2 = app.clone().oneshot(req2).await.unwrap();
        let body2 = resp2.into_body().collect().await.unwrap().to_bytes();
        let json2: serde_json::Value = serde_json::from_slice(&body2).unwrap();
        assert_eq!(json2["layer"], 1);

        let req3 = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .header("x-thread-id", "thread-b")
            .body(json_body("hello"))
            .unwrap();
        let resp3 = app.oneshot(req3).await.unwrap();
        let body3 = resp3.into_body().collect().await.unwrap().to_bytes();
        let json3: serde_json::Value = serde_json::from_slice(&body3).unwrap();
        assert_eq!(json3["layer"], 3);

        let session_a = derive_session_cache_scope("thread-a");
        let session_b = derive_session_cache_scope("thread-b");
        let key_a = build_exact_cache_key("native", "hello\nmodel: test", session_a.as_deref());
        let key_b = build_exact_cache_key("native", "hello\nmodel: test", session_b.as_deref());

        assert!(state.exact_cache.get(&key_a).is_some());
        assert!(state.exact_cache.get(&key_b).is_some());
    }

    #[tokio::test]
    async fn exact_cache_uses_body_session_identifier_when_header_missing() {
        let state = test_state(CacheMode::Exact);
        let app = cache_app(state.clone());
        let body = serde_json::to_vec(&serde_json::json!({
            "prompt": "hello",
            "metadata": {"session_id": "body-session"}
        }))
        .unwrap();

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let session = derive_session_cache_scope("body-session");
        let scoped_key = build_exact_cache_key("native", "hello\nmodel: test", session.as_deref());
        let global_key = build_exact_cache_key("native", "hello\nmodel: test", None);

        assert!(state.exact_cache.get(&scoped_key).is_some());
        assert!(state.exact_cache.get(&global_key).is_none());
    }
}

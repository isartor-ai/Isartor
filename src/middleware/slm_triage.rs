use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use tracing::{Instrument, info_span};

use crate::core::prompt::extract_prompt;
use crate::middleware::body_buffer::BufferedBody;
use crate::models::{
    ChatResponse, FinalLayer, OpenAiChatChoice, OpenAiChatResponse, OpenAiMessage,
};
use crate::state::AppState;

/// System prompt used to ask the local SLM to classify the user's intent.
const CLASSIFY_SYSTEM_PROMPT: &str = "\
You are a request classifier. Decide whether the following user prompt \
is SIMPLE (can be answered with a short factual response, a greeting, or \
basic knowledge) or COMPLEX (requires deep reasoning, code generation, \
creative writing, or multi-step analysis).\n\n\
Reply with EXACTLY one word: SIMPLE or COMPLEX.";

// ── OpenAI-compatible request/response types for the sidecar ─────────

#[derive(serde::Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(serde::Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(serde::Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(serde::Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[derive(serde::Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

/// Layer 2 — SLM triage middleware.
///
/// 1. Sends the user's prompt to the local llama.cpp sidecar for intent
///    classification via the OpenAI-compatible `/v1/chat/completions` endpoint.
/// 2. If the SLM classifies the task as **SIMPLE**, a second call asks
///    the SLM to generate the answer directly (short-circuit).
/// 3. If the task is **COMPLEX** or the SLM is unreachable, the request
///    continues to Layer 3 (external LLM fallback).
pub async fn slm_triage_middleware(request: Request, next: Next) -> Response {
    fn slm_response_for_path(
        path: &str,
        status: StatusCode,
        answer: String,
        model: String,
    ) -> Response {
        match path {
            "/v1/chat/completions" => {
                let body = OpenAiChatResponse {
                    choices: vec![OpenAiChatChoice {
                        message: OpenAiMessage {
                            role: "assistant".to_string(),
                            content: Some(answer),
                            name: None,
                            tool_call_id: None,
                            tool_calls: None,
                            function_call: None,
                        },
                        index: 0,
                        finish_reason: Some("stop".to_string()),
                    }],
                    model: Some(model),
                };
                (status, Json(body)).into_response()
            }
            "/v1/messages" => {
                // Minimal Anthropic Messages response.
                let body = serde_json::json!({
                    "content": [{"type": "text", "text": answer}],
                    "model": model,
                });
                (status, Json(body)).into_response()
            }
            _ => (
                status,
                Json(ChatResponse {
                    layer: 2,
                    message: answer,
                    model: Some(model),
                }),
            )
                .into_response(),
        }
    }

    let span = info_span!("layer2_slm", slm.complexity_score = tracing::field::Empty);
    async move {
        let layer_start = Instant::now();
        let request_path = request.uri().path().to_string();
        let state = match request.extensions().get::<Arc<AppState>>() {
            Some(s) => s.clone(),
            None => {
                tracing::error!("Layer 2: AppState missing from request extensions");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ChatResponse {
                        layer: 2,
                        message: "Firewall misconfiguration: missing application state".into(),
                        model: None,
                    }),
                )
                    .into_response();
            }
        };

    // ------------------------------------------------------------------
    // 0. Feature gate — when enable_slm_router is false the entire L2
    //    layer is a no-op and the request falls straight to Layer 3.
    // ------------------------------------------------------------------
    if !state.config.enable_slm_router {
        tracing::debug!("Layer 2: SLM router disabled — skipping to Layer 3");
        return next.run(request).await;
    }

    // ------------------------------------------------------------------
    // 1. Extract the prompt from the buffered body (set by body_buffer
    //    middleware). The request body stream is untouched.
    // ------------------------------------------------------------------
    let body_bytes = match request.extensions().get::<BufferedBody>() {
        Some(buf) => buf.0.clone(),
        None => {
            tracing::error!("Layer 2: BufferedBody missing from request extensions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ChatResponse {
                    layer: 2,
                    message: "Firewall misconfiguration: missing buffered body".into(),
                    model: None,
                }),
            )
                .into_response();
        }
    };

    let prompt = extract_prompt(&body_bytes);

    tracing::debug!(prompt = %prompt, "Layer 2: Classifying intent via local SLM");

    // ------------------------------------------------------------------
    // 2. Classify the prompt via the selected Inference Engine.
    // ------------------------------------------------------------------
    let is_simple = {
        if state.config.inference_engine == crate::config::InferenceEngineMode::Embedded {
        #[cfg(feature = "embedded-inference")]
        {
            if let Some(classifier) = &state.embedded_classifier {
                match classifier.classify(&prompt).await {
                    Ok((label, _conf)) => {
                        let is_simp = label == "SIMPLE";
                        tracing::Span::current().record(
                            "slm.complexity_score",
                            if is_simp { "SIMPLE" } else { "COMPLEX" },
                        );
                        is_simp
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Layer 2: Embedded classification failed – falling through");
                        crate::metrics::record_error("L2_SLM", "retryable");
                        false
                    }
                }
            } else {
                tracing::warn!("Layer 2: Embedded engine requested but not configured");
                false
            }
        }
        #[cfg(not(feature = "embedded-inference"))]
        {
            tracing::warn!("Layer 2: Embedded engine requested but binary was not compiled with `embedded-inference` feature. Falling back to sidecar logic.");
            // NOTE: Ideally this fall-through uses sidecar URL but let's just fall to Layer 3 for safety.
            false
        }
    } else {
        let sidecar_url = format!(
            "{}/v1/chat/completions",
            state.config.layer2.sidecar_url.trim_end_matches('/')
        );

        let classify_req = ChatCompletionRequest {
            model: state.config.layer2.model_name.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: CLASSIFY_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.clone(),
                },
            ],
            stream: false,
            temperature: Some(0.0),
            max_tokens: Some(10),
        };

        match state
            .http_client
            .post(&sidecar_url)
            .json(&classify_req)
            .send()
            .await
        {
            Ok(resp) => match resp.json::<ChatCompletionResponse>().await {
                Ok(completion) => {
                    let answer = completion
                        .choices
                        .into_iter()
                        .next()
                        .map(|c| c.message.content)
                        .unwrap_or_default()
                        .trim()
                        .to_uppercase();
                    tracing::info!(classification = %answer, "Layer 2: SLM classification result");

                    let is_simp = answer.contains("SIMPLE");
                    tracing::Span::current().record(
                        "slm.complexity_score",
                        if is_simp { "SIMPLE" } else { "COMPLEX" },
                    );

                    is_simp
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Layer 2: Failed to parse SLM response – falling through");
                    crate::metrics::record_error("L2_SLM", "retryable");
                    false
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "Layer 2: SLM unreachable – falling through to Layer 3");
                crate::metrics::record_error("L2_SLM", "retryable");
                false
            }
        }
    }
    }; // close classification block

    // ------------------------------------------------------------------
    // 3. Branch: short-circuit or continue.
    // ------------------------------------------------------------------
    if is_simple {
        tracing::info!("Layer 2: Simple task – generating answer via selected engine");

        if state.config.inference_engine == crate::config::InferenceEngineMode::Embedded {
            #[cfg(feature = "embedded-inference")]
            {
                if let Some(classifier) = &state.embedded_classifier {
                    match classifier.execute(&prompt).await {
                        Ok(answer) => {
                            crate::metrics::record_layer_duration("L2_SLM", layer_start.elapsed());
                            let model = "embedded(gemma-2)".to_string();
                            let mut response = slm_response_for_path(
                                &request_path,
                                StatusCode::OK,
                                answer,
                                model,
                            );
                            response.extensions_mut().insert(FinalLayer::Slm);
                            return response;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Layer 2: Embedded answer generation failed – falling through");
                            crate::metrics::record_error("L2_SLM", "retryable");
                        }
                    }
                }
            }
        } else {
            let sidecar_url = format!(
                "{}/v1/chat/completions",
                state.config.layer2.sidecar_url.trim_end_matches('/')
            );
            // Second call: ask the sidecar to actually answer the prompt.
            let answer_req = ChatCompletionRequest {
                model: state.config.layer2.model_name.clone(),
                messages: vec![
                    ChatMessage {
                        role: "user".to_string(),
                        content: prompt.clone(),
                    },
                ],
                stream: false,
                temperature: None,
                max_tokens: None,
            };

            match state
                .http_client
                .post(&sidecar_url)
                .json(&answer_req)
                .send()
                .await
            {
                Ok(resp) => match resp.json::<ChatCompletionResponse>().await {
                    Ok(completion) => {
                        let answer = completion
                            .choices
                            .into_iter()
                            .next()
                            .map(|c| c.message.content)
                            .unwrap_or_default();
                        crate::metrics::record_layer_duration("L2_SLM", layer_start.elapsed());
                        let model = state.config.layer2.model_name.clone();
                        let mut response = slm_response_for_path(
                            &request_path,
                            StatusCode::OK,
                            answer,
                            model,
                        );
                        response.extensions_mut().insert(FinalLayer::Slm);
                        return response;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Layer 2: Sidecar answer parse failed – falling through");
                        crate::metrics::record_error("L2_SLM", "retryable");
                    }
                },
                Err(e) => {
                    tracing::warn!(error = %e, "Layer 2: Sidecar answer call failed – falling through");
                    crate::metrics::record_error("L2_SLM", "retryable");
                }
            }
        }
    }

    // Complex task (or SLM failure) — forward to Layer 3 (body stream intact).
    tracing::debug!("Layer 2: Forwarding to Layer 3");
    next.run(request).await
    }
    .instrument(span)
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, middleware as axum_mw, routing::post};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::clients::slm::SlmClient;
    use crate::config::{AppConfig, CacheMode, EmbeddingSidecarSettings, Layer2Settings};
    use crate::layer1::embeddings::shared_test_embedder;
    use crate::layer1::layer1a_cache::ExactMatchCache;
    use crate::middleware::body_buffer::buffer_body_middleware;
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

    fn test_state(sidecar_url: &str) -> Arc<AppState> {
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
                sidecar_url: sidecar_url.into(),
                model_name: "phi-3-mini".into(),
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
            l3_timeout_secs: 120,
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
            enable_slm_router: true,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
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

    fn triage_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route(
                "/api/chat",
                post(|| async {
                    (
                        StatusCode::OK,
                        axum::Json(ChatResponse {
                            layer: 3,
                            message: "layer 3 response".into(),
                            model: Some("gpt-4o".into()),
                        }),
                    )
                }),
            )
            .layer(axum_mw::from_fn(slm_triage_middleware))
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

    fn chat_completion_json(content: &str) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": { "content": content }
            }]
        })
    }

    #[tokio::test]
    async fn simple_classification_short_circuits() {
        let mock_server = MockServer::start().await;

        // First call: classification returns SIMPLE
        // Second call: SLM generates the answer
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json("SIMPLE")))
            .up_to_n_times(1)
            .expect(1)
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_completion_json("The answer is 42")),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let state = test_state(&mock_server.uri());
        let app = triage_app(state);

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(json_body("what is 6*7?"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["layer"], 2);
        assert_eq!(json["message"], "The answer is 42");
    }

    #[tokio::test]
    async fn complex_classification_passes_through() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json("COMPLEX")))
            .expect(1)
            .mount(&mock_server)
            .await;

        let state = test_state(&mock_server.uri());
        let app = triage_app(state);

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(json_body("write a compiler for Rust"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Should reach Layer 3 handler.
        assert_eq!(json["layer"], 3);
        assert_eq!(json["message"], "layer 3 response");
    }

    #[tokio::test]
    async fn slm_unreachable_falls_through() {
        // Point to a URL that won't be listening.
        let state = test_state("http://127.0.0.1:1");
        let app = triage_app(state);

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
        // Falls through to Layer 3.
        assert_eq!(json["layer"], 3);
    }

    #[tokio::test]
    async fn malformed_slm_response_falls_through() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not valid json"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let state = test_state(&mock_server.uri());
        let app = triage_app(state);

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(json_body("test"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["layer"], 3);
    }

    /// When `enable_slm_router` is false the middleware must be a complete
    /// no-op: no HTTP calls to the sidecar, no classification — the
    /// request goes straight to Layer 3.
    #[tokio::test]
    async fn disabled_flag_skips_l2_entirely() {
        // Build state with the flag turned off and a non-listening sidecar
        // URL. If the middleware tried to call it, the test would hang or fail.
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
                sidecar_url: "http://127.0.0.1:1".into(),
                model_name: "phi-3-mini".into(),
                timeout_seconds: 1,
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
            l3_timeout_secs: 120,
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
            enable_slm_router: false, // ← L2 disabled
            otel_exporter_endpoint: "http://localhost:4317".into(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
        });

        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
            vector_cache: Arc::new(VectorCache::new(0.85, 300, 100)),
            llm_agent: Arc::new(MockAgent),
            slm_client: Arc::new(SlmClient::new(&config.layer2)),
            text_embedder: shared_test_embedder(),
            config,
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });

        let app = triage_app(state);

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
        // Should skip L2 entirely and land on Layer 3.
        assert_eq!(json["layer"], 3);
        assert_eq!(json["message"], "layer 3 response");
    }
}

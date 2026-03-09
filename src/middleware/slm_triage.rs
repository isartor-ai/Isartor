use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use http_body_util::BodyExt;
use tracing::{info_span, Instrument};

use crate::models::ChatResponse;
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
    let span = info_span!("layer2_slm", slm.complexity_score = tracing::field::Empty);
    async move {
        let state = request
            .extensions()
            .get::<Arc<AppState>>()
            .expect("AppState missing from request extensions")
            .clone();

    // ------------------------------------------------------------------
    // 1. Read the request body.
    // ------------------------------------------------------------------
    let (parts, body) = request.into_parts();
    let body_bytes: Bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ChatResponse {
                    layer: 2,
                    message: "Failed to read request body".into(),
                    model: None,
                }),
            )
                .into_response();
        }
    };

    // Try to parse as JSON `{ "prompt": "..." }`, fall back to raw string.
    let prompt: String = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("prompt").and_then(|p| p.as_str()).map(String::from))
        .unwrap_or_else(|| String::from_utf8_lossy(&body_bytes).to_string());

    tracing::debug!(prompt = %prompt, "Layer 2: Classifying intent via local SLM");

    // ------------------------------------------------------------------
    // 2. Classify the prompt via the llama.cpp sidecar.
    // ------------------------------------------------------------------
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

    let classification = state
        .http_client
        .post(&sidecar_url)
        .json(&classify_req)
        .send()
        .await;

    let is_simple = match classification {
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
                false
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "Layer 2: SLM unreachable – falling through to Layer 3");
            false
        }
    };

    // ------------------------------------------------------------------
    // 3. Branch: short-circuit or continue.
    // ------------------------------------------------------------------
    if is_simple {
        tracing::info!("Layer 2: Simple task – generating answer via local SLM");

        // Second call: ask the SLM to actually answer the prompt.
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
                    return (
                        StatusCode::OK,
                        Json(ChatResponse {
                            layer: 2,
                            message: answer,
                            model: Some(state.config.layer2.model_name.clone()),
                        }),
                    )
                        .into_response();
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Layer 2: SLM answer parse failed – falling through");
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "Layer 2: SLM answer call failed – falling through");
            }
        }
    }

    // Complex task (or SLM failure) — re-attach body, forward to Layer 3.
    tracing::debug!("Layer 2: Forwarding to Layer 3");
    let request = Request::from_parts(parts, Body::from(body_bytes));
    next.run(request).await
    }
    .instrument(span)
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{middleware as axum_mw, routing::post, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::clients::slm::SlmClient;
    use crate::config::{AppConfig, CacheMode, EmbeddingSidecarSettings, Layer2Settings};
    use crate::models::ChatResponse;
    use crate::state::{AppLlmAgent, ExactCache};
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

    fn test_state(sidecar_url: &str) -> Arc<AppState> {
        let config = Arc::new(AppConfig {
            host_port: "127.0.0.1:0".into(),
            gateway_api_key: "test".into(),
            cache_mode: CacheMode::Exact,
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
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
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
            exact_cache: Arc::new(ExactCache::new(300, 100)),
            vector_cache: Arc::new(VectorCache::new(0.85, 300, 100)),
            llm_agent: Arc::new(MockAgent),
            slm_client: Arc::new(SlmClient::new(&config.layer2)),
            config,
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
}

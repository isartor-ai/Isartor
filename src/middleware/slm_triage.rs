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

use crate::core::prompt::extract_classifier_context;
use crate::middleware::body_buffer::BufferedBody;
use crate::models::{
    ChatResponse, FinalLayer, OpenAiChatChoice, OpenAiChatResponse, OpenAiMessage,
    OpenAiMessageContent,
};
use crate::state::AppState;

/// System prompt used to ask the local SLM to classify the user's intent.
const CLASSIFY_SYSTEM_PROMPT: &str = "\
You are a request classifier. Decide whether the following user prompt \
is SIMPLE (can be answered with a short factual response, a greeting, or \
basic knowledge) or COMPLEX (requires deep reasoning, code generation, \
creative writing, or multi-step analysis).\n\n\
Reply with EXACTLY one word: SIMPLE or COMPLEX.";

/// Tiered classifier prompt that recognises code-adjacent tasks a small
/// model can handle locally (config files, type definitions, short
/// snippets) without routing everything labelled "code" to the cloud.
pub const CLASSIFY_TIERED_SYSTEM_PROMPT: &str = "\
You are a coding task classifier for a local small language model.\n\
Classify the prompt into exactly one category:\n\n\
TEMPLATE — The task produces a configuration file (tsconfig, package.json, \
Dockerfile, docker-compose, jest.config, .gitignore, .env, ESLint config), \
a TypeScript/JavaScript type or interface definition, or documentation \
(README, JSDoc, code comments). Output is predictable and template-like.\n\n\
SNIPPET — The task produces a single short code file or function (under 50 lines), \
such as a server entry point, a simple validation function, a basic middleware, \
or a data type with minimal logic. Does NOT reference or import multiple \
project-specific modules.\n\n\
COMPLEX — The task requires generating substantial code (over 50 lines), implements \
full CRUD endpoints, writes test suites, builds frontend pages, imports from \
multiple project files, or requires understanding project-wide architecture.\n\n\
Reply with EXACTLY one word: TEMPLATE, SNIPPET, or COMPLEX.";

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

/// Answer quality guard — rejects SLM answers that are likely too poor to
/// serve.  Falls through to L3 when the answer looks empty, uncertain, or
/// suspiciously short.
pub(crate) fn answer_quality_ok(answer: &str) -> bool {
    let trimmed = answer.trim();
    if trimmed.len() < 10 {
        return false;
    }
    let lower = trimmed.to_lowercase();
    let uncertainty = [
        "i don't know",
        "i'm not sure",
        "i cannot",
        "i can't",
        "as an ai",
        "i am unable",
    ];
    if uncertainty.iter().any(|u| lower.starts_with(u)) {
        return false;
    }
    true
}

/// Layer 2 — SLM triage middleware.
///
/// 1. Sends the user's prompt to the local llama.cpp sidecar for intent
///    classification via the OpenAI-compatible `/v1/chat/completions` endpoint.
/// 2. In **tiered** mode the classifier labels the prompt as TEMPLATE,
///    SNIPPET, or COMPLEX.  TEMPLATE and SNIPPET are deflected to L2.
///    In **binary** mode (legacy) the labels are SIMPLE / COMPLEX.
/// 3. If the task is deflectable, a second call asks the SLM to generate
///    the answer directly (short-circuit).
/// 4. If the task is COMPLEX, the SLM is unreachable, or the answer
///    fails the quality guard, the request continues to Layer 3.
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
                            content: Some(OpenAiMessageContent::text(answer)),
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
        let tool = crate::tool_identity::identify_tool_or_fallback(
            request
                .headers()
                .get(axum::http::header::USER_AGENT)
                .and_then(|value| value.to_str().ok()),
            "gateway",
        );
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

    let prompt = extract_classifier_context(&body_bytes);

    tracing::debug!(prompt = %prompt, "Layer 2: Classifying intent via local SLM");

    // ------------------------------------------------------------------
    // 2. Classify the prompt via the selected Inference Engine.
    //    `is_deflectable` is true when the SLM can answer locally.
    //    In tiered mode: TEMPLATE or SNIPPET → deflectable.
    //    In binary mode: SIMPLE → deflectable.
    // ------------------------------------------------------------------
    let use_tiered = state.config.layer2.classifier_mode
        == crate::config::ClassifierMode::Tiered;

    let is_deflectable = {
        if state.config.inference_engine == crate::config::InferenceEngineMode::Embedded {
        #[cfg(feature = "embedded-inference")]
        {
            if let Some(classifier) = &state.embedded_classifier {
                match classifier.classify(&prompt).await {
                    Ok((label, _conf)) => {
                        let deflect = if use_tiered {
                            label == "TEMPLATE" || label == "SNIPPET" || label == "SIMPLE"
                        } else {
                            label == "SIMPLE"
                        };
                        tracing::Span::current().record(
                            "slm.complexity_score",
                            &label as &str,
                        );
                        deflect
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Layer 2: Embedded classification failed – falling through");
                        crate::metrics::record_error_with_tool("L2_SLM", "retryable", tool);
                        crate::visibility::record_agent_error(tool);
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
            false
        }
    } else {
        // Classification uses local_slm (CPU-friendly, always-on).
        // Strip the Ollama generate suffix if present to get the OpenAI-compat base URL.
        let classify_base = state
            .config
            .local_slm_url
            .trim_end_matches("/api/generate")
            .trim_end_matches('/');
        let classify_url = format!("{classify_base}/v1/chat/completions");

        let system_prompt = if use_tiered {
            CLASSIFY_TIERED_SYSTEM_PROMPT
        } else {
            CLASSIFY_SYSTEM_PROMPT
        };

        let classify_req = ChatCompletionRequest {
            model: state.config.local_slm_model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
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

        let classify_timeout =
            std::time::Duration::from_secs(state.config.layer2.timeout_seconds as u64);

        tracing::debug!(
            classify_url = %classify_url,
            classify_model = %state.config.local_slm_model,
            "Layer 2: Sending classification to local SLM"
        );

        match state
            .http_client
            .post(&classify_url)
            .timeout(classify_timeout)
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

                    let deflect = if use_tiered {
                        answer.contains("TEMPLATE") || answer.contains("SNIPPET")
                    } else {
                        answer.contains("SIMPLE")
                    };
                    tracing::Span::current().record(
                        "slm.complexity_score",
                        answer.as_str(),
                    );

                    deflect
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Layer 2: Failed to parse classifier response – falling through");
                    crate::metrics::record_error_with_tool("L2_SLM", "retryable", tool);
                    crate::visibility::record_agent_error(tool);
                    false
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "Layer 2: Classifier unreachable – falling through to L3");
                crate::metrics::record_error_with_tool("L2_SLM", "retryable", tool);
                crate::visibility::record_agent_error(tool);
                false
            }
        }
    }
    }; // close classification block

    // ------------------------------------------------------------------
    // 3. Branch: short-circuit or continue.
    // ------------------------------------------------------------------
    if is_deflectable {
        tracing::info!("Layer 2: Deflectable task – generating answer via selected engine");

        if state.config.inference_engine == crate::config::InferenceEngineMode::Embedded {
            #[cfg(feature = "embedded-inference")]
            {
                if let Some(classifier) = &state.embedded_classifier {
                    match classifier.execute(&prompt).await {
                        Ok(answer) => {
                            // Answer quality guard
                            if answer_quality_ok(&answer) {
                                crate::metrics::record_layer_duration_with_tool(
                                    "L2_SLM",
                                    layer_start.elapsed(),
                                    tool,
                                );
                                let model = "embedded(gemma-2)".to_string();
                                let mut response = slm_response_for_path(
                                    &request_path,
                                    StatusCode::OK,
                                    answer,
                                    model,
                                );
                                response.extensions_mut().insert(FinalLayer::Slm);
                                return response;
                            } else {
                                tracing::info!("Layer 2: Answer quality guard rejected SLM output – falling through to L3");
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Layer 2: Embedded answer generation failed – falling through");
                            crate::metrics::record_error_with_tool("L2_SLM", "retryable", tool);
                            crate::visibility::record_agent_error(tool);
                        }
                    }
                }
            }
        } else {
            // Answer generation uses the GPU sidecar (layer2.sidecar_url).
            let sidecar_url = format!(
                "{}/v1/chat/completions",
                state.config.layer2.sidecar_url.trim_end_matches('/')
            );
            let max_tokens = state.config.layer2.max_answer_tokens;
            let gen_timeout =
                std::time::Duration::from_secs(state.config.layer2.timeout_seconds as u64);

            tracing::debug!(
                generation_url = %sidecar_url,
                generation_model = %state.config.layer2.model_name,
                "Layer 2: Sending answer generation to GPU sidecar"
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
                max_tokens: Some(max_tokens),
            };

            match state
                .http_client
                .post(&sidecar_url)
                .timeout(gen_timeout)
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
                        // Answer quality guard
                        if answer_quality_ok(&answer) {
                            crate::metrics::record_layer_duration_with_tool(
                                "L2_SLM",
                                layer_start.elapsed(),
                                tool,
                            );
                            let model = state.config.layer2.model_name.clone();
                            let mut response = slm_response_for_path(
                                &request_path,
                                StatusCode::OK,
                                answer,
                                model,
                            );
                            response.extensions_mut().insert(FinalLayer::Slm);
                            return response;
                        } else {
                            tracing::info!("Layer 2: Answer quality guard rejected SLM output – falling through to L3");
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Layer 2: Sidecar answer parse failed – falling through");
                        crate::metrics::record_error_with_tool("L2_SLM", "retryable", tool);
                        crate::visibility::record_agent_error(tool);
                    }
                },
                Err(e) => {
                    tracing::warn!(error = %e, "Layer 2: Sidecar answer call failed – falling through");
                    crate::metrics::record_error_with_tool("L2_SLM", "retryable", tool);
                    crate::visibility::record_agent_error(tool);
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

    use crate::config::{AppConfig, ClassifierMode};
    use crate::middleware::body_buffer::buffer_body_middleware;
    use crate::models::ChatResponse;
    use crate::state::AppLlmAgent;

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
        let mut cfg = AppConfig::test_default();
        cfg.enable_slm_router = true;
        cfg.layer2.sidecar_url = sidecar_url.into();
        // Point local_slm_url to the same mock server so classification
        // requests (which now use local_slm_url) reach the mock.
        cfg.local_slm_url = sidecar_url.into();
        let config = Arc::new(cfg);
        AppState::test_with_agent(Arc::new(MockAgent), config)
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

    #[test]
    fn answer_quality_guard_accepts_valid_answers() {
        assert!(answer_quality_ok(
            "The answer is 42, because 6 times 7 equals 42."
        ));
        assert!(answer_quality_ok("function hello() { return 42; }"));
        assert!(answer_quality_ok(
            "{ \"compilerOptions\": { \"strict\": true } }"
        ));
    }

    #[test]
    fn answer_quality_guard_rejects_poor_answers() {
        assert!(!answer_quality_ok(""));
        assert!(!answer_quality_ok("   "));
        assert!(!answer_quality_ok("short"));
        assert!(!answer_quality_ok("I don't know how to do that."));
        assert!(!answer_quality_ok("I'm not sure about that."));
        assert!(!answer_quality_ok("I cannot help with that."));
        assert!(!answer_quality_ok("As an AI language model, I cannot..."));
    }

    #[tokio::test]
    async fn simple_classification_short_circuits() {
        let mock_server = MockServer::start().await;

        // First call: classification returns TEMPLATE (tiered mode default)
        // Second call: SLM generates the answer
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_completion_json("TEMPLATE")),
            )
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
    async fn snippet_classification_short_circuits() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json("SNIPPET")))
            .up_to_n_times(1)
            .expect(1)
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completion_json("function hello() { return 42; }")),
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
            .body(json_body("write a hello function"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["layer"], 2);
    }

    #[tokio::test]
    async fn binary_mode_still_uses_simple_complex() {
        let mock_server = MockServer::start().await;

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

        // Build state with binary classifier mode
        let mut cfg = AppConfig::test_default();
        cfg.enable_slm_router = true;
        cfg.layer2.sidecar_url = mock_server.uri();
        cfg.local_slm_url = mock_server.uri();
        cfg.layer2.classifier_mode = ClassifierMode::Binary;
        let config = Arc::new(cfg);
        let state = AppState::test_with_agent(Arc::new(MockAgent), config);

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
        let mut cfg = AppConfig::test_default();
        cfg.enable_slm_router = false; // ← L2 disabled
        cfg.layer2.sidecar_url = "http://127.0.0.1:1".into();
        cfg.layer2.timeout_seconds = 1;
        let config = Arc::new(cfg);
        let state = AppState::test_with_agent(Arc::new(MockAgent), config);

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

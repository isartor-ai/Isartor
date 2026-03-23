use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use axum::response::IntoResponse;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{Instrument, info_span};

use crate::anthropic_sse;
use crate::config::LlmProvider;
use crate::core::prompt::{extract_prompt, has_tooling};
use crate::core::retry::{RetryConfig, execute_with_retry};
use crate::errors::GatewayError;
use crate::middleware::body_buffer::BufferedBody;
use crate::models::{
    ChatResponse, FinalLayer, OpenAiChatChoice, OpenAiChatRequest, OpenAiChatResponse,
    OpenAiMessage, OpenAiModel, OpenAiModelList,
};
use crate::providers::copilot::exchange_copilot_session_token;
use crate::state::AppState;
use crate::visibility;

fn configured_openai_models(state: &AppState) -> OpenAiModelList {
    let provider = state.config.llm_provider.as_str();
    let model_id = match state.config.llm_provider {
        crate::config::LlmProvider::Azure if !state.config.azure_deployment_id.is_empty() => {
            state.config.azure_deployment_id.clone()
        }
        _ => state.config.external_llm_model.clone(),
    };

    OpenAiModelList::new(vec![OpenAiModel::new(model_id, provider)])
}

fn configured_openai_model_id(state: &AppState) -> String {
    configured_openai_models(state)
        .data
        .into_iter()
        .next()
        .map(|model| model.id)
        .unwrap_or_else(|| state.config.external_llm_model.clone())
}

fn supports_openai_passthrough(provider: &LlmProvider) -> bool {
    matches!(
        provider,
        LlmProvider::Openai
            | LlmProvider::Azure
            | LlmProvider::Copilot
            | LlmProvider::Xai
            | LlmProvider::Mistral
            | LlmProvider::Groq
            | LlmProvider::Deepseek
            | LlmProvider::Galadriel
            | LlmProvider::Hyperbolic
            | LlmProvider::Moonshot
            | LlmProvider::Openrouter
            | LlmProvider::Perplexity
            | LlmProvider::Together
    )
}

fn provider_chat_completions_url(state: &AppState) -> Option<String> {
    match &state.config.llm_provider {
        LlmProvider::Azure => Some(format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            state.config.external_llm_url.trim_end_matches('/'),
            state.config.azure_deployment_id,
            state.config.azure_api_version
        )),
        LlmProvider::Copilot => Some(if state.config.external_llm_url.trim().is_empty() {
            "https://api.githubcopilot.com/chat/completions".to_string()
        } else {
            state.config.external_llm_url.clone()
        }),
        provider if supports_openai_passthrough(provider) => {
            Some(state.config.external_llm_url.clone())
        }
        _ => None,
    }
}

async fn send_openai_passthrough_request(
    state: &AppState,
    request: &OpenAiChatRequest,
) -> anyhow::Result<String> {
    let Some(url) = provider_chat_completions_url(state) else {
        anyhow::bail!(
            "provider {} does not support OpenAI tool passthrough",
            state.config.llm_provider
        );
    };

    let mut payload = serde_json::to_value(request)?;
    if let Value::Object(ref mut map) = payload {
        map.insert(
            "model".to_string(),
            Value::String(configured_openai_model_id(state)),
        );
    }

    let mut request_builder = state
        .http_client
        .post(url)
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/json");

    match state.config.llm_provider {
        LlmProvider::Azure => {
            request_builder = request_builder.header("api-key", &state.config.external_llm_api_key);
        }
        LlmProvider::Copilot => {
            let copilot_token = exchange_copilot_session_token(
                &state.http_client,
                &state.config.external_llm_api_key,
            )
            .await?;
            request_builder = request_builder
                .header(AUTHORIZATION, format!("Bearer {copilot_token}"))
                .header("User-Agent", "GitHubCopilotChat/0.29.1")
                .header("Editor-Version", "vscode/1.99.0")
                .header("Editor-Plugin-Version", "copilot-chat/0.29.1")
                .header("Copilot-Integration-Id", "vscode-chat")
                .header("X-GitHub-Api-Version", "2025-04-01");
        }
        _ => {
            request_builder = request_builder.header(
                AUTHORIZATION,
                format!("Bearer {}", state.config.external_llm_api_key),
            );
        }
    }

    let response = request_builder.json(&payload).send().await?;
    let status = response.status();
    let body = response.text().await?;

    if !status.is_success() {
        anyhow::bail!("HTTP {status}: {body}");
    }

    Ok(body)
}

/// Layer 3 — Fallback handler.
///
/// Runs **only** if every preceding middleware layer decided it could
/// not handle the request. Dispatches the prompt to the configured
/// LLM provider via `rig-core`.
///
/// When `offline_mode` is `true` the handler immediately returns HTTP 503
/// rather than attempting any outbound cloud connection.
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
                        message: "Firewall misconfiguration: missing application state".into(),
                        model: None,
                    }),
                )
                    .into_response();
            }
        };

        // ------------------------------------------------------------------
        // 0. Offline mode guard — immediately reject L3 cloud calls.
        // ------------------------------------------------------------------
        if state.config.offline_mode {
            tracing::warn!(
                "Layer 3: request blocked — ISARTOR__OFFLINE_MODE=true"
            );
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "offline_mode_active",
                    "message": "This request could not be resolved locally. \
                                Cloud routing is disabled in offline mode.",
                    "layer_reached": "L3",
                    "suggestion": "Lower your semantic similarity threshold \
                                   (ISARTOR__SIMILARITY_THRESHOLD) to increase \
                                   local deflection rate."
                })),
            )
                .into_response();
        }

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
                    message: "Firewall misconfiguration: missing buffered body".into(),
                    model: None,
                }),
            )
                .into_response();
        }
    };

    let prompt = extract_prompt(&body_bytes);

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
            // Cache keys are now namespaced by endpoint format (e.g. "native|<prompt>")
            // to prevent cross-endpoint schema poisoning. For stale fallback, we try
            // the new namespaced key first, then fall back to the legacy key for
            // backwards compatibility with older cache entries.
            let legacy_key = hex::encode(Sha256::digest(prompt.as_bytes()));
            let namespaced_input = format!("native|{prompt}");
            let namespaced_key = hex::encode(Sha256::digest(namespaced_input.as_bytes()));

            for exact_key in [namespaced_key, legacy_key] {
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

/// OpenAI-compatible chat completions endpoint — `POST /v1/chat/completions`.
///
/// This is used by many agent frameworks and SDKs that expect an OpenAI-style API.
pub async fn openai_chat_completions_handler(request: Request) -> impl IntoResponse {
    let span = info_span!("openai_chat_completions");
    async move {
        let layer_start = Instant::now();
        let state = match request.extensions().get::<Arc<AppState>>() {
            Some(s) => s.clone(),
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {"message": "missing application state"}
                    })),
                )
                    .into_response();
            }
        };

        if state.config.offline_mode {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {"message": "offline mode active"}
                })),
            )
                .into_response();
        }

        let body_bytes = match request.extensions().get::<BufferedBody>() {
            Some(buf) => buf.0.clone(),
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {"message": "missing buffered body"}
                    })),
                )
                    .into_response();
            }
        };

        let provider_name = state.llm_agent.provider_name();
        tracing::info!(provider = provider_name, "OpenAI compat: forwarding to LLM");

        if has_tooling(&body_bytes) {
            let request = match serde_json::from_slice::<OpenAiChatRequest>(&body_bytes) {
                Ok(request) => request,
                Err(err) => {
                    let mut resp = (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": {"message": format!("invalid OpenAI request body: {err}")}
                        })),
                    )
                        .into_response();
                    resp.extensions_mut().insert(FinalLayer::Cloud);
                    return resp;
                }
            };

            let retry_cfg = RetryConfig::default();
            let provider_for_err = provider_name.to_string();
            let state_for_retry = state.clone();
            let request_for_retry = request.clone();
            let result = execute_with_retry(&retry_cfg, "L3_OpenAIToolPassthrough", || {
                let state = state_for_retry.clone();
                let request = request_for_retry.clone();
                let provider = provider_for_err.clone();
                async move {
                    send_openai_passthrough_request(&state, &request)
                        .await
                        .map_err(|e| GatewayError::from_llm_error(&provider, &e))
                }
            })
            .await;

            match result {
                Ok(body) => {
                    crate::metrics::record_layer_duration("L3_Cloud", layer_start.elapsed());
                    let mut resp = (StatusCode::OK, [(CONTENT_TYPE, "application/json")], body)
                        .into_response();
                    resp.extensions_mut().insert(FinalLayer::Cloud);
                    return resp;
                }
                Err(gw_err) => {
                    crate::metrics::record_layer_duration("L3_Cloud", layer_start.elapsed());
                    let mut resp = (
                        StatusCode::BAD_GATEWAY,
                        Json(serde_json::json!({
                            "error": {"message": format!("[{provider_name}] {gw_err}")}
                        })),
                    )
                        .into_response();
                    resp.extensions_mut().insert(FinalLayer::Cloud);
                    return resp;
                }
            }
        }

        let prompt = extract_prompt(&body_bytes);

        let retry_cfg = RetryConfig::default();
        let agent = state.llm_agent.clone();
        let provider_for_err = provider_name.to_string();
        let prompt_for_retry = prompt.clone();

        let result = execute_with_retry(&retry_cfg, "L3_OpenAICompat", || {
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

                let response = OpenAiChatResponse {
                    choices: vec![OpenAiChatChoice {
                        message: OpenAiMessage {
                            role: "assistant".to_string(),
                            content: Some(text),
                            name: None,
                            tool_call_id: None,
                            tool_calls: None,
                            function_call: None,
                        },
                        index: 0,
                        finish_reason: Some("stop".to_string()),
                    }],
                    model: Some(state.config.external_llm_model.clone()),
                };

                let mut resp = (StatusCode::OK, Json(response)).into_response();
                resp.extensions_mut().insert(FinalLayer::Cloud);
                resp
            }
            Err(gw_err) => {
                crate::metrics::record_layer_duration("L3_Cloud", layer_start.elapsed());
                let mut resp = (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({
                        "error": {"message": format!("[{provider_name}] {gw_err}")}
                    })),
                )
                    .into_response();
                resp.extensions_mut().insert(FinalLayer::Cloud);
                resp
            }
        }
    }
    .instrument(span)
    .await
}

/// OpenAI-compatible models endpoint — `GET /v1/models`.
pub async fn openai_models_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {"message": "missing application state"}
                })),
            )
                .into_response();
        }
    };

    (StatusCode::OK, Json(configured_openai_models(&state))).into_response()
}

/// Anthropic Messages endpoint — `POST /v1/messages`.
///
/// Used by Claude Code and other Anthropic-compatible clients.
pub async fn anthropic_messages_handler(request: Request) -> impl IntoResponse {
    let span = info_span!("anthropic_messages");
    async move {
        let layer_start = Instant::now();
        let state = match request.extensions().get::<Arc<AppState>>() {
            Some(s) => s.clone(),
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {"message": "missing application state"}
                    })),
                )
                    .into_response();
            }
        };

        if state.config.offline_mode {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {"message": "offline mode active"}
                })),
            )
                .into_response();
        }

        let body_bytes = match request.extensions().get::<BufferedBody>() {
            Some(buf) => buf.0.clone(),
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {"message": "missing buffered body"}
                    })),
                )
                    .into_response();
            }
        };

        let prompt = extract_prompt(&body_bytes);

        let provider_name = state.llm_agent.provider_name();
        tracing::info!(
            provider = provider_name,
            "Anthropic compat: forwarding to LLM"
        );

        let retry_cfg = RetryConfig::default();
        let agent = state.llm_agent.clone();
        let provider_for_err = provider_name.to_string();
        let prompt_for_retry = prompt.clone();

        let result = execute_with_retry(&retry_cfg, "L3_AnthropicCompat", || {
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

        let model = &state.config.external_llm_model;

        match result {
            Ok(text) => {
                crate::metrics::record_layer_duration("L3_Cloud", layer_start.elapsed());
                let mut resp = (
                    StatusCode::OK,
                    Json(anthropic_sse::build_json_response(&text, model)),
                )
                    .into_response();
                resp.extensions_mut().insert(FinalLayer::Cloud);
                resp
            }
            Err(gw_err) => {
                crate::metrics::record_layer_duration("L3_Cloud", layer_start.elapsed());
                let mut resp = (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("[{provider_name}] {gw_err}")
                        }
                    })),
                )
                    .into_response();
                resp.extensions_mut().insert(FinalLayer::Cloud);
                resp
            }
        }
    }
    .instrument(span)
    .await
}

/// Copilot CLI preToolUse hook endpoint.
///
/// Called by the Copilot CLI hook script before each tool use.
/// For v0.1: logs the tool call and returns `action: "allow"`.
/// Future versions will add tool-call caching and policy enforcement.
pub async fn pretooluse_hook_handler(body: axum::body::Bytes) -> impl IntoResponse {
    #[derive(serde::Deserialize)]
    struct PreToolUseRequest {
        #[serde(default)]
        tool: String,
        #[serde(default)]
        args: String,
        #[serde(default)]
        #[allow(dead_code)]
        timestamp: String,
    }

    let parsed: PreToolUseRequest = serde_json::from_slice(&body).unwrap_or(PreToolUseRequest {
        tool: "unknown".into(),
        args: String::new(),
        timestamp: String::new(),
    });

    tracing::info!(
        tool = %parsed.tool,
        args_len = parsed.args.len(),
        "Copilot CLI tool call intercepted"
    );

    Json(serde_json::json!({
        "action": "allow",
        "reason": null,
        "result": null,
        "cached": false,
        "logged": true,
    }))
}

/// Cache lookup endpoint for MCP / external clients.
///
/// Checks L1a (exact) and L1b (semantic) caches without hitting L3.
/// Returns the cached response if found, or 204 No Content on miss.
pub async fn cache_lookup_handler(request: Request) -> impl IntoResponse {
    let started_at = Instant::now();
    let state = request
        .extensions()
        .get::<Arc<AppState>>()
        .cloned()
        .expect("AppState missing");

    // Extract User-Agent before consuming the request body.
    let user_agent = request
        .headers()
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let tool = crate::tool_identity::identify_tool_or_fallback(
        if user_agent.is_empty() {
            None
        } else {
            Some(&user_agent)
        },
        "mcp",
    );

    let body_bytes = match axum::body::to_bytes(request.into_body(), 1024 * 64).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid body"})),
            )
                .into_response();
        }
    };

    let prompt = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("prompt").and_then(|p| p.as_str()).map(String::from))
        .unwrap_or_default();

    if prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt is required"})),
        )
            .into_response();
    }

    // L1a exact match.
    let namespaced = format!("native|{prompt}");
    let exact_key = hex::encode(Sha256::digest(namespaced.as_bytes()));
    if let Some(cached) = state.exact_cache.get(&exact_key) {
        tracing::info!("Cache lookup: L1a exact hit");
        record_cache_lookup_prompt(
            "l1a",
            true,
            &prompt,
            started_at.elapsed(),
            StatusCode::OK,
            tool,
        );
        return (
            StatusCode::OK,
            [
                ("X-Isartor-Layer", "l1a"),
                ("X-Isartor-Deflected", "true"),
                ("Content-Type", "application/json"),
            ],
            cached,
        )
            .into_response();
    }

    // L1b semantic match.
    let embedder = state.text_embedder.clone();
    let prompt_for_embed = prompt.clone();
    if let Ok(Ok(embedding)) =
        tokio::task::spawn_blocking(move || embedder.generate_embedding(&prompt_for_embed)).await
        && let Some(cached) = state.vector_cache.search(&embedding).await
    {
        tracing::info!("Cache lookup: L1b semantic hit");
        record_cache_lookup_prompt(
            "l1b",
            true,
            &prompt,
            started_at.elapsed(),
            StatusCode::OK,
            tool,
        );
        return (
            StatusCode::OK,
            [
                ("X-Isartor-Layer", "l1b"),
                ("X-Isartor-Deflected", "true"),
                ("Content-Type", "application/json"),
            ],
            cached,
        )
            .into_response();
    }

    // Cache miss.
    record_cache_lookup_prompt(
        "miss",
        false,
        &prompt,
        started_at.elapsed(),
        StatusCode::NO_CONTENT,
        tool,
    );
    (StatusCode::NO_CONTENT).into_response()
}

/// Cache store endpoint for MCP / external clients.
///
/// Stores a prompt/response pair in L1a (exact) and L1b (semantic) caches.
/// Used by MCP tools to cache responses after the client's own LLM answers.
pub async fn cache_store_handler(request: Request) -> impl IntoResponse {
    let state = request
        .extensions()
        .get::<Arc<AppState>>()
        .cloned()
        .expect("AppState missing");

    let body_bytes = match axum::body::to_bytes(request.into_body(), 1024 * 256).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid body"})),
            )
                .into_response();
        }
    };

    #[derive(serde::Deserialize)]
    struct CacheStoreRequest {
        prompt: String,
        response: String,
        #[serde(default)]
        model: String,
    }

    let req: CacheStoreRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "prompt and response are required"})),
            )
                .into_response();
        }
    };

    // Build a ChatResponse to store (normalized to layer 1).
    let cached_json = serde_json::to_string(&ChatResponse {
        layer: 1,
        message: req.response.clone(),
        model: Some(if req.model.is_empty() {
            "cached".to_string()
        } else {
            req.model
        }),
    })
    .unwrap_or_default();

    // L1a exact cache.
    let namespaced = format!("native|{}", req.prompt);
    let exact_key = hex::encode(Sha256::digest(namespaced.as_bytes()));
    state.exact_cache.put(exact_key, cached_json.clone());

    // L1b semantic cache.
    let embedder = state.text_embedder.clone();
    let prompt_for_embed = req.prompt.clone();
    if let Ok(Ok(embedding)) =
        tokio::task::spawn_blocking(move || embedder.generate_embedding(&prompt_for_embed)).await
    {
        state.vector_cache.insert(embedding, cached_json).await;
    }

    tracing::info!(
        prompt_len = req.prompt.len(),
        "Cache store: written to L1a+L1b"
    );

    (StatusCode::OK, Json(serde_json::json!({"stored": true}))).into_response()
}

fn record_cache_lookup_prompt(
    final_layer: &str,
    deflected: bool,
    prompt: &str,
    elapsed: std::time::Duration,
    status_code: StatusCode,
    tool: &str,
) {
    crate::metrics::record_request_with_tool(
        final_layer,
        status_code.as_u16(),
        elapsed.as_secs_f64(),
        "mcp",
        "copilot",
        "cache_lookup",
        tool,
    );

    if deflected {
        crate::metrics::record_tokens_saved_with_tool(
            final_layer,
            crate::metrics::estimate_tokens(prompt),
            "mcp",
            "copilot",
            "cache_lookup",
            tool,
        );
    }

    visibility::record_prompt(crate::models::PromptVisibilityEntry {
        timestamp: chrono::Utc::now().to_rfc3339(),
        traffic_surface: "mcp".to_string(),
        client: "copilot".to_string(),
        endpoint_family: "cache_lookup".to_string(),
        route: "/api/v1/cache/lookup".to_string(),
        prompt_hash: Some(hex::encode(Sha256::digest(prompt.as_bytes()))),
        final_layer: final_layer.to_string(),
        resolved_by: if deflected {
            None
        } else {
            Some("copilot_upstream".to_string())
        },
        deflected,
        latency_ms: elapsed.as_millis() as u64,
        status_code: status_code.as_u16(),
        tool: tool.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        middleware as axum_mw,
        routing::{get, post},
    };
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
    use crate::visibility::{clear_prompt_stats, prompt_stats_snapshot};
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
            l3_timeout_secs: 120,
            azure_deployment_id: "".into(),
            azure_api_version: "".into(),
            enable_monitoring: false,
            enable_slm_router: false,
            otel_exporter_endpoint: "http://localhost:4317".into(),
            offline_mode: false,
            proxy_port: "0.0.0.0:8081".into(),
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
            .route(
                "/v1/chat/completions",
                post(openai_chat_completions_handler),
            )
            .route("/v1/models", get(openai_models_handler))
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

    fn cache_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/api/v1/cache/lookup", post(cache_lookup_handler))
            .route("/api/v1/cache/store", post(cache_store_handler))
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
        assert!(
            json["message"]
                .as_str()
                .unwrap()
                .contains("provider outage")
        );
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
    async fn openai_non_tool_request_preserves_text_only_behavior() {
        let state = test_state(Arc::new(SuccessAgent));
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({
            "model": "ignored-by-isartor",
            "messages": [{"role": "user", "content": "hello openai"}]
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(
            json["choices"][0]["message"]["content"],
            "Reply to: user: hello openai"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["model"], "gpt-4o-mini");
    }

    #[tokio::test]
    async fn openai_tool_request_passthrough_preserves_tool_calls() {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{body_partial_json, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "model": "gpt-4o-mini",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup_weather"
                    }
                }],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "lookup_weather"}
                },
                "functions": [{
                    "name": "legacy_lookup"
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": "{\"city\":\"Berlin\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "model": "gpt-4o-mini"
            })))
            .mount(&server)
            .await;

        let state = test_state(Arc::new(SuccessAgent));
        let mut config = (*state.config).clone();
        config.external_llm_url = format!("{}/v1/chat/completions", server.uri());

        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            config: Arc::new(config),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({
            "model": "client-specified-model",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_prev",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_prev", "content": "{\"temp_c\":21}"},
                {"role": "user", "content": "What next?"}
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
            },
            "functions": [{
                "name": "legacy_lookup",
                "parameters": {"type": "object"}
            }]
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Berlin\"}"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn openai_models_returns_configured_l3_model_list() {
        let state = test_state(Arc::new(SuccessAgent));
        let app = handler_app(state);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "gpt-4o-mini");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["owned_by"], "openai");
    }

    #[tokio::test]
    async fn openai_models_prefers_azure_deployment_id_when_configured() {
        let state = test_state(Arc::new(SuccessAgent));
        let mut config = (*state.config).clone();
        config.llm_provider = crate::config::LlmProvider::Azure;
        config.azure_deployment_id = "azure-deployment".into();
        config.external_llm_model = "gpt-4o-mini".into();

        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            config: Arc::new(config),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });
        let app = handler_app(state);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["data"][0]["id"], "azure-deployment");
        assert_eq!(json["data"][0]["owned_by"], "azure");
    }

    #[tokio::test]
    async fn stale_cache_fallback_on_llm_failure() {
        let state = test_state(Arc::new(FailAgent));

        // Pre-populate the exact cache with a stale entry for "fallback test".
        let prompt = "fallback test";
        let key_input = format!("native|{prompt}");
        let key = hex::encode(Sha256::digest(key_input.as_bytes()));
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

    #[tokio::test]
    async fn cache_lookup_hit_is_recorded_in_prompt_stats() {
        clear_prompt_stats();
        let state = test_state(Arc::new(SuccessAgent));
        let app = cache_app(state);

        let store_req = Request::builder()
            .method("POST")
            .uri("/api/v1/cache/store")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "prompt": "capital of France",
                    "response": "Paris."
                }))
                .unwrap(),
            ))
            .unwrap();
        let store_resp = app.clone().oneshot(store_req).await.unwrap();
        assert_eq!(store_resp.status(), StatusCode::OK);

        let lookup_req = Request::builder()
            .method("POST")
            .uri("/api/v1/cache/lookup")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "prompt": "capital of France"
                }))
                .unwrap(),
            ))
            .unwrap();
        let lookup_resp = app.oneshot(lookup_req).await.unwrap();
        assert_eq!(lookup_resp.status(), StatusCode::OK);

        let stats = prompt_stats_snapshot(50);
        assert!(stats.by_surface.get("mcp").copied().unwrap_or(0) >= 1);
        assert!(stats.by_client.get("copilot").copied().unwrap_or(0) >= 1);
        assert!(stats.by_layer.get("l1a").copied().unwrap_or(0) >= 1);
        assert!(stats.recent.iter().any(|entry| {
            entry.traffic_surface == "mcp"
                && entry.client == "copilot"
                && entry.final_layer == "l1a"
                && entry.route == "/api/v1/cache/lookup"
        }));
    }

    #[tokio::test]
    async fn cache_lookup_miss_is_recorded_in_prompt_stats() {
        clear_prompt_stats();
        let state = test_state(Arc::new(SuccessAgent));
        let app = cache_app(state);

        let lookup_req = Request::builder()
            .method("POST")
            .uri("/api/v1/cache/lookup")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "prompt": "not cached yet"
                }))
                .unwrap(),
            ))
            .unwrap();
        let lookup_resp = app.oneshot(lookup_req).await.unwrap();
        assert_eq!(lookup_resp.status(), StatusCode::NO_CONTENT);

        let stats = prompt_stats_snapshot(50);
        assert!(stats.by_surface.get("mcp").copied().unwrap_or(0) >= 1);
        assert!(stats.by_client.get("copilot").copied().unwrap_or(0) >= 1);
        assert!(stats.by_layer.get("miss").copied().unwrap_or(0) >= 1);
        assert!(stats.recent.iter().any(|entry| {
            entry.traffic_surface == "mcp"
                && entry.client == "copilot"
                && entry.final_layer == "miss"
                && entry.route == "/api/v1/cache/lookup"
                && entry.resolved_by.as_deref() == Some("copilot_upstream")
        }));
    }
}

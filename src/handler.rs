use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axum::Json;
use axum::body::Body;
use axum::extract::Request;
use axum::http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::IntervalStream;
use tracing::{Instrument, info_span};

use crate::anthropic_sse;
use crate::config::{
    DEFAULT_OPENAI_CHAT_COMPLETIONS_URL, LlmProvider, default_chat_completions_url,
};
use crate::core::cache_scope::{
    build_exact_cache_key, extract_session_cache_scope, namespaced_semantic_cache_input,
};
use crate::core::prompt::{
    extract_prompt, extract_request_model, extract_route_model, has_tooling,
    is_gemini_streaming_path, override_request_model,
};
use crate::core::quota::evaluate_provider_quota;
use crate::core::retry::{RetryConfig, execute_with_retry};
use crate::errors::GatewayError;
use crate::gemini_sse;
use crate::mcp::{self, ToolExecutor};
use crate::middleware::body_buffer::BufferedBody;
use crate::models::{
    ChatResponse, FinalLayer, OpenAiChatChoice, OpenAiChatRequest, OpenAiChatResponse,
    OpenAiMessage, OpenAiMessageContent, OpenAiModel, OpenAiModelList,
};
use crate::providers::copilot::exchange_copilot_session_token;
use crate::state::{AppState, ResolvedProviderConfig};
use crate::visibility;

fn configured_openai_models(state: &AppState) -> OpenAiModelList {
    let provider = state.config.llm_provider.as_str();
    let mut data = Vec::new();
    let mut seen = HashSet::new();
    let default_model = state.config.configured_model_id();

    if seen.insert(default_model.clone()) {
        data.push(OpenAiModel::new(default_model, provider));
    }

    let mut alias_entries = state
        .config
        .model_aliases
        .iter()
        .map(|(alias, target)| (alias.clone(), target.clone()))
        .collect::<Vec<_>>();
    alias_entries.sort_by(|a, b| a.0.cmp(&b.0));

    for (_, target) in &alias_entries {
        if seen.insert(target.clone()) {
            data.push(OpenAiModel::new(target.clone(), provider));
        }
    }

    for (alias, _) in alias_entries {
        if seen.insert(alias.clone()) {
            data.push(OpenAiModel::new(alias, format!("alias:{provider}")));
        }
    }

    OpenAiModelList::new(data)
}

fn configured_openai_model_id(state: &AppState) -> String {
    state.config.configured_model_id()
}

fn resolved_request_model_for_path(state: &AppState, body_bytes: &[u8], path: &str) -> String {
    extract_request_model(body_bytes)
        .or_else(|| extract_route_model(path))
        .map(|model| state.config.resolve_model_alias(&model))
        .unwrap_or_else(|| configured_openai_model_id(state))
}

fn canonicalize_request_body_model(
    state: &AppState,
    body_bytes: &[u8],
    path: &str,
) -> (Vec<u8>, String) {
    let resolved_model = resolved_request_model_for_path(state, body_bytes, path);
    (
        override_request_model(body_bytes, &resolved_model),
        resolved_model,
    )
}

fn gemini_error_response(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "code": status.as_u16(),
                "message": message,
                "status": status
                    .canonical_reason()
                    .unwrap_or("UNKNOWN")
                    .to_ascii_uppercase()
                    .replace(' ', "_")
            }
        })),
    )
        .into_response()
}

fn request_tool(request: &Request, traffic_surface: &str) -> &'static str {
    crate::tool_identity::identify_tool_or_fallback(
        request
            .headers()
            .get(axum::http::header::USER_AGENT)
            .and_then(|value| value.to_str().ok()),
        traffic_surface,
    )
}

fn record_provider_success(state: &AppState, provider: &ResolvedProviderConfig) {
    state.record_provider_success(provider);
}

fn record_provider_failure(
    state: &AppState,
    provider: &ResolvedProviderConfig,
    error: &impl std::fmt::Display,
) {
    state.record_provider_failure(provider, &error.to_string());
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
            | LlmProvider::Cerebras
            | LlmProvider::Nebius
            | LlmProvider::Siliconflow
            | LlmProvider::Fireworks
            | LlmProvider::Nvidia
            | LlmProvider::Chutes
            | LlmProvider::Deepseek
            | LlmProvider::Galadriel
            | LlmProvider::Hyperbolic
            | LlmProvider::Moonshot
            | LlmProvider::Openrouter
            | LlmProvider::Perplexity
            | LlmProvider::Together
    )
}

fn provider_chat_completions_url(provider: &ResolvedProviderConfig) -> Option<String> {
    match &provider.provider {
        LlmProvider::Azure => Some(format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            provider.endpoint.trim_end_matches('/'),
            provider.azure_deployment_id,
            provider.azure_api_version
        )),
        LlmProvider::Copilot => Some(if provider.endpoint.trim().is_empty() {
            "https://api.githubcopilot.com/chat/completions".to_string()
        } else {
            provider.endpoint.clone()
        }),
        provider_kind if supports_openai_passthrough(provider_kind) => {
            let configured_url = provider.endpoint.trim();
            let default_url = default_chat_completions_url(provider_kind)?;

            if configured_url.is_empty()
                || (*provider_kind != LlmProvider::Openai
                    && configured_url == DEFAULT_OPENAI_CHAT_COMPLETIONS_URL)
            {
                Some(default_url.to_string())
            } else {
                Some(configured_url.to_string())
            }
        }
        _ => None,
    }
}

async fn send_openai_passthrough_request(
    state: &AppState,
    provider: &ResolvedProviderConfig,
    request: &OpenAiChatRequest,
) -> anyhow::Result<String> {
    let selected_key = state.provider_key_pools.acquire(provider)?;
    let execution_provider = provider.with_api_key(selected_key.api_key.clone());

    let Some(url) = provider_chat_completions_url(&execution_provider) else {
        anyhow::bail!(
            "provider {} does not support OpenAI tool passthrough",
            provider.provider_name()
        );
    };

    let mut payload = serde_json::to_value(request)?;
    if let Value::Object(ref mut map) = payload {
        map.insert("model".to_string(), Value::String(request.model.clone()));
    }

    let mut request_builder = state
        .http_client
        .post(url)
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/json");

    match execution_provider.provider {
        LlmProvider::Azure => {
            request_builder = request_builder.header("api-key", &execution_provider.api_key);
        }
        LlmProvider::Copilot => {
            let copilot_token =
                exchange_copilot_session_token(&state.http_client, &execution_provider.api_key)
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
                format!("Bearer {}", execution_provider.api_key),
            );
        }
    }

    let response = request_builder.json(&payload).send().await?;
    let status = response.status();
    let body = response.text().await?;

    if !status.is_success() {
        state.provider_key_pools.record_result(
            provider,
            &execution_provider.api_key,
            Some(&format!("HTTP {status}: {body}")),
        );
        anyhow::bail!("HTTP {status}: {body}");
    }

    state
        .provider_key_pools
        .record_result(provider, &execution_provider.api_key, None);

    Ok(body)
}

struct ProviderExecutionResult {
    provider: ResolvedProviderConfig,
    model: String,
    body: String,
}

fn gateway_error_status(error: &GatewayError) -> StatusCode {
    match error {
        GatewayError::Quota { .. } => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::BAD_GATEWAY,
    }
}

fn add_provider_header(response: &mut Response, provider: &ResolvedProviderConfig) {
    if let Ok(value) = HeaderValue::from_str(provider.provider_name()) {
        response.headers_mut().insert("x-isartor-provider", value);
    }
}

/// Check provider quota and return either a blocking error or `Ok(())`.
fn check_provider_quota(
    state: &AppState,
    provider: &ResolvedProviderConfig,
    projected_total_tokens: u64,
    projected_prompt_tokens: u64,
    projected_completion_tokens: u64,
) -> Result<(), GatewayError> {
    let projected_cost_usd = crate::core::usage::estimate_event_cost_usd(
        &state.config,
        provider.provider_name(),
        projected_prompt_tokens,
        projected_completion_tokens,
    );
    if let Some(quota_decision) = evaluate_provider_quota(
        &state.config,
        &state.usage_tracker,
        provider.provider_name(),
        projected_total_tokens,
        projected_cost_usd,
        chrono::Utc::now(),
    ) {
        let provider_name = provider.provider_name();
        for warning in quota_decision.warning_messages {
            tracing::warn!(
                provider = %provider_name,
                action = ?quota_decision.action,
                "{warning}"
            );
        }

        if let Some(limit_message) = quota_decision.limit_message {
            return Err(GatewayError::Quota {
                provider: provider_name.to_string(),
                message: limit_message,
                fallback_allowed: quota_decision.action
                    == crate::config::QuotaLimitAction::Fallback,
            });
        }
    }
    Ok(())
}

/// Handle a provider call result: on success return the execution result,
/// on fallback-eligible failure record and continue, otherwise return the error.
fn handle_provider_result(
    state: &AppState,
    provider: &ResolvedProviderConfig,
    model: String,
    result: Result<String, GatewayError>,
    last_error: &mut Option<GatewayError>,
) -> Option<Result<ProviderExecutionResult, GatewayError>> {
    match result {
        Ok(body) => Some(Ok(ProviderExecutionResult {
            provider: provider.clone(),
            model,
            body,
        })),
        Err(error) => {
            record_provider_failure(state, provider, &error);
            if error.should_fallback_to_next_provider() {
                *last_error = Some(error);
                None
            } else {
                Some(Err(error))
            }
        }
    }
}

async fn execute_prompt_provider_chain(
    state: Arc<AppState>,
    prompt: String,
    primary_model: String,
    operation: &'static str,
    tool: &'static str,
) -> Result<ProviderExecutionResult, GatewayError> {
    let mut last_error = None;
    let projected_total_tokens = crate::metrics::estimate_tokens(&prompt);
    let projected_prompt_tokens = crate::metrics::estimate_prompt_tokens(&prompt);
    let projected_completion_tokens =
        projected_total_tokens.saturating_sub(projected_prompt_tokens);

    for provider in state.provider_chain.iter() {
        if let Err(e) = check_provider_quota(
            &state,
            provider,
            projected_total_tokens,
            projected_prompt_tokens,
            projected_completion_tokens,
        ) {
            record_provider_failure(&state, provider, &e);
            if e.should_fallback_to_next_provider() {
                last_error = Some(e);
                continue;
            }
            return Err(e);
        }

        let model = if provider.active {
            primary_model.clone()
        } else {
            provider.configured_model_id().to_string()
        };
        let retry_cfg = RetryConfig::cloud_llm();
        let state_c = state.clone();
        let prompt_c = prompt.clone();
        let provider_c = provider.clone();
        let model_c = model.clone();
        let provider_name = provider.provider_name().to_string();
        let result = execute_with_retry(&retry_cfg, operation, tool, move || {
            let state = state_c.clone();
            let prompt = prompt_c.clone();
            let provider = provider_c.clone();
            let model = model_c.clone();
            let provider_name = provider_name.clone();
            async move {
                state
                    .chat_with_provider(&provider, &prompt, Some(&model))
                    .await
                    .map_err(|e| GatewayError::from_llm_error(&provider_name, &e))
            }
        })
        .await;

        if let Some(outcome) =
            handle_provider_result(&state, provider, model, result, &mut last_error)
        {
            return outcome;
        }
    }

    Err(last_error.unwrap_or_else(|| GatewayError::Configuration {
        message: "no Layer 3 providers configured".to_string(),
    }))
}

async fn execute_passthrough_provider_chain(
    state: Arc<AppState>,
    request: OpenAiChatRequest,
    tool: &'static str,
) -> Result<ProviderExecutionResult, GatewayError> {
    let mut last_error = None;
    let request_body = serde_json::to_vec(&request).unwrap_or_default();
    let passthrough_prompt = extract_prompt(&request_body);
    let projected_total_tokens = crate::metrics::estimate_tokens(&passthrough_prompt);
    let projected_prompt_tokens = crate::metrics::estimate_prompt_tokens(&passthrough_prompt);
    let projected_completion_tokens =
        projected_total_tokens.saturating_sub(projected_prompt_tokens);

    for provider in state.provider_chain.iter() {
        if !supports_openai_passthrough(&provider.provider) {
            continue;
        }

        if let Err(e) = check_provider_quota(
            &state,
            provider,
            projected_total_tokens,
            projected_prompt_tokens,
            projected_completion_tokens,
        ) {
            record_provider_failure(&state, provider, &e);
            if e.should_fallback_to_next_provider() {
                last_error = Some(e);
                continue;
            }
            return Err(e);
        }

        let retry_cfg = RetryConfig::cloud_llm();
        let state_c = state.clone();
        let request_c = request.clone();
        let provider_c = provider.clone();
        let provider_name = provider.provider_name().to_string();
        let model = provider.configured_model_id().to_string();
        let result = execute_with_retry(&retry_cfg, "L3_OpenAIToolPassthrough", tool, move || {
            let state = state_c.clone();
            let request = request_c.clone();
            let provider = provider_c.clone();
            let provider_name = provider_name.clone();
            async move {
                send_openai_passthrough_request(&state, &provider, &request)
                    .await
                    .map_err(|e| GatewayError::from_llm_error(&provider_name, &e))
            }
        })
        .await;

        if let Some(outcome) =
            handle_provider_result(&state, provider, model, result, &mut last_error)
        {
            return outcome;
        }
    }

    Err(last_error.unwrap_or_else(|| GatewayError::Configuration {
        message: "no Layer 3 providers support OpenAI tool passthrough".to_string(),
    }))
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
        let tool = request_tool(&request, "gateway");
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

        let (canonical_body, resolved_model) = canonicalize_request_body_model(&state, &body_bytes, request.uri().path());
        let session_cache_scope = extract_session_cache_scope(request.headers(), &canonical_body);
        let prompt = extract_prompt(&canonical_body);
        let cache_key_material = crate::core::prompt::extract_cache_key(&canonical_body);

        tracing::Span::current().record("ai.prompt.length_bytes", prompt.len() as u64);

        let provider_name = state.primary_provider().provider_name();
        tracing::Span::current().record("provider.name", provider_name);
        tracing::Span::current().record("model", resolved_model.as_str());
        tracing::info!(prompt = %prompt, provider = provider_name, "Layer 3: Forwarding to LLM via Rig");

        let result = execute_prompt_provider_chain(
            state.clone(),
            prompt.clone(),
            resolved_model.clone(),
            "L3_Cloud_LLM",
            tool,
        )
        .await;

        match result {
            Ok(result) => {
                record_provider_success(&state, &result.provider);
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );
                let mut response = (
                    StatusCode::OK,
                        Json(ChatResponse {
                            layer: 3,
                            message: result.body,
                            model: Some(result.model),
                        }),
                )
                    .into_response();
                add_provider_header(&mut response, &result.provider);
                response.extensions_mut().insert(FinalLayer::Cloud);
                response
            }
            Err(gw_err) => {
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );
                crate::metrics::record_error_with_tool(
                    gw_err.layer_label(),
                    if gw_err.is_retryable() {
                        "retryable"
                    } else {
                        "fatal"
                    },
                    tool,
                );
                crate::visibility::record_agent_error(tool);
                tracing::error!(error = %gw_err, provider = provider_name, "Layer 3: LLM call failed after retries");

                // ── Stale-cache fallback ─────────────────────────────
                // If the LLM is down, try to serve a previously-cached
                // answer for this exact prompt so the user still gets
                // *something* useful.
                // Cache keys are now namespaced by endpoint format (e.g. "native|<prompt>")
                // to prevent cross-endpoint schema poisoning. For stale fallback, we try
                // the new namespaced key first, then fall back to the legacy key for
                // backwards compatibility with older cache entries.
                let exact_keys = if session_cache_scope.is_some() {
                    vec![build_exact_cache_key(
                        "native",
                        &cache_key_material,
                        session_cache_scope.as_deref(),
                    )]
                } else {
                    vec![
                        build_exact_cache_key("native", &cache_key_material, None),
                        hex::encode(Sha256::digest(prompt.as_bytes())),
                    ]
                };

                for exact_key in exact_keys {
                    if let Some(cached) = state.exact_cache.get(&exact_key) {
                        tracing::info!(
                            cache.key = %exact_key,
                            "Layer 3: Serving stale cache entry as fallback"
                        );
                        crate::metrics::record_error_with_tool(
                            "L3_StaleFallback",
                            "fallback_used",
                            tool,
                        );
                        crate::visibility::record_agent_error(tool);
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
                    gateway_error_status(&gw_err),
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
        let tool = request_tool(&request, "gateway");
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

        let (canonical_body, resolved_model) =
            canonicalize_request_body_model(&state, &body_bytes, request.uri().path());
        let provider_name = state.primary_provider().provider_name();
        tracing::info!(provider = provider_name, "OpenAI compat: forwarding to LLM");

        if has_tooling(&canonical_body) {
            let request = match serde_json::from_slice::<OpenAiChatRequest>(&canonical_body) {
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

            let result = execute_passthrough_provider_chain(state.clone(), request, tool).await;

            match result {
                Ok(result) => {
                    record_provider_success(&state, &result.provider);
                    crate::metrics::record_layer_duration_with_tool(
                        "L3_Cloud",
                        layer_start.elapsed(),
                        tool,
                    );
                    let mut resp = (
                        StatusCode::OK,
                        [(CONTENT_TYPE, "application/json")],
                        result.body,
                    )
                        .into_response();
                    add_provider_header(&mut resp, &result.provider);
                    resp.extensions_mut().insert(FinalLayer::Cloud);
                    return resp;
                }
                Err(gw_err) => {
                    crate::metrics::record_layer_duration_with_tool(
                        "L3_Cloud",
                        layer_start.elapsed(),
                        tool,
                    );
                    crate::metrics::record_error_with_tool(
                        gw_err.layer_label(),
                        if gw_err.is_retryable() {
                            "retryable"
                        } else {
                            "fatal"
                        },
                        tool,
                    );
                    crate::visibility::record_agent_error(tool);
                    let mut resp = (
                        gateway_error_status(&gw_err),
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

        let prompt = extract_prompt(&canonical_body);

        let result = execute_prompt_provider_chain(
            state.clone(),
            prompt.clone(),
            resolved_model.clone(),
            "L3_OpenAICompat",
            tool,
        )
        .await;

        match result {
            Ok(result) => {
                record_provider_success(&state, &result.provider);
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );

                let response = OpenAiChatResponse {
                    choices: vec![OpenAiChatChoice {
                        message: OpenAiMessage {
                            role: "assistant".to_string(),
                            content: Some(OpenAiMessageContent::text(result.body)),
                            name: None,
                            tool_call_id: None,
                            tool_calls: None,
                            function_call: None,
                        },
                        index: 0,
                        finish_reason: Some("stop".to_string()),
                    }],
                    model: Some(result.model),
                };

                let mut resp = (StatusCode::OK, Json(response)).into_response();
                add_provider_header(&mut resp, &result.provider);
                resp.extensions_mut().insert(FinalLayer::Cloud);
                resp
            }
            Err(gw_err) => {
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );
                crate::metrics::record_error_with_tool(
                    gw_err.layer_label(),
                    if gw_err.is_retryable() {
                        "retryable"
                    } else {
                        "fatal"
                    },
                    tool,
                );
                crate::visibility::record_agent_error(tool);
                let mut resp = (
                    gateway_error_status(&gw_err),
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

pub async fn provider_status_handler(request: Request) -> impl IntoResponse {
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

    (StatusCode::OK, Json(state.provider_status())).into_response()
}

/// Gemini-native GenerateContent endpoints.
pub async fn gemini_generate_content_handler(request: Request) -> impl IntoResponse {
    let span = info_span!("gemini_generate_content");
    async move {
        let layer_start = Instant::now();
        let tool = request_tool(&request, "gateway");
        let state = match request.extensions().get::<Arc<AppState>>() {
            Some(s) => s.clone(),
            None => {
                return gemini_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "missing application state".to_string(),
                );
            }
        };

        if state.config.offline_mode {
            return gemini_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "offline mode active".to_string(),
            );
        }

        let body_bytes = match request.extensions().get::<BufferedBody>() {
            Some(buf) => buf.0.clone(),
            None => {
                return gemini_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "missing buffered body".to_string(),
                );
            }
        };

        let (canonical_body, resolved_model) =
            canonicalize_request_body_model(&state, &body_bytes, request.uri().path());
        let prompt = extract_prompt(&canonical_body);
        let provider_name = state.primary_provider().provider_name();
        let is_streaming = is_gemini_streaming_path(request.uri().path());

        let result = execute_prompt_provider_chain(
            state.clone(),
            prompt,
            resolved_model.clone(),
            if is_streaming {
                "L3_GeminiStreamCompat"
            } else {
                "L3_GeminiCompat"
            },
            tool,
        )
        .await;

        match result {
            Ok(result) => {
                record_provider_success(&state, &result.provider);
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );
                let mut response = if is_streaming {
                    gemini_sse::build_sse_response(&result.body, &result.model)
                } else {
                    (
                        StatusCode::OK,
                        Json(gemini_sse::build_json_response(&result.body, &result.model)),
                    )
                        .into_response()
                };
                add_provider_header(&mut response, &result.provider);
                response.extensions_mut().insert(FinalLayer::Cloud);
                response
            }
            Err(gw_err) => {
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );
                crate::metrics::record_error_with_tool(
                    gw_err.layer_label(),
                    if gw_err.is_retryable() {
                        "retryable"
                    } else {
                        "fatal"
                    },
                    tool,
                );
                crate::visibility::record_agent_error(tool);
                let mut response = gemini_error_response(
                    gateway_error_status(&gw_err),
                    format!("[{provider_name}] {gw_err}"),
                );
                response.extensions_mut().insert(FinalLayer::Cloud);
                response
            }
        }
    }
    .instrument(span)
    .await
}

/// Anthropic Messages endpoint — `POST /v1/messages`.
///
/// Used by Claude Code and other Anthropic-compatible clients.
pub async fn anthropic_messages_handler(request: Request) -> impl IntoResponse {
    let span = info_span!("anthropic_messages");
    async move {
        let layer_start = Instant::now();
        let tool = request_tool(&request, "gateway");
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

        let (canonical_body, resolved_model) =
            canonicalize_request_body_model(&state, &body_bytes, request.uri().path());
        let prompt = extract_prompt(&canonical_body);

        let provider_name = state.primary_provider().provider_name();
        tracing::info!(
            provider = provider_name,
            "Anthropic compat: forwarding to LLM"
        );

        let result = execute_prompt_provider_chain(
            state.clone(),
            prompt.clone(),
            resolved_model.clone(),
            "L3_AnthropicCompat",
            tool,
        )
        .await;

        match result {
            Ok(result) => {
                record_provider_success(&state, &result.provider);
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );
                let mut resp = (
                    StatusCode::OK,
                    Json(anthropic_sse::build_json_response(
                        &result.body,
                        &result.model,
                    )),
                )
                    .into_response();
                add_provider_header(&mut resp, &result.provider);
                resp.extensions_mut().insert(FinalLayer::Cloud);
                resp
            }
            Err(gw_err) => {
                crate::metrics::record_layer_duration_with_tool(
                    "L3_Cloud",
                    layer_start.elapsed(),
                    tool,
                );
                crate::metrics::record_error_with_tool(
                    gw_err.layer_label(),
                    if gw_err.is_retryable() {
                        "retryable"
                    } else {
                        "fatal"
                    },
                    tool,
                );
                crate::visibility::record_agent_error(tool);
                let mut resp = (
                    gateway_error_status(&gw_err),
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

    let (parts, body) = request.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 1024 * 64).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid body"})),
            )
                .into_response();
        }
    };
    let session_cache_scope = extract_session_cache_scope(&parts.headers, &body_bytes);

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
    let exact_key = build_exact_cache_key("native", &prompt, session_cache_scope.as_deref());
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
    let prompt_for_embed = namespaced_semantic_cache_input("native", &prompt);
    if let Ok(Ok(embedding)) =
        tokio::task::spawn_blocking(move || embedder.generate_embedding(&prompt_for_embed)).await
        && let Some(cached) = state
            .vector_cache
            .search(&embedding, session_cache_scope.as_deref())
            .await
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

    let (parts, body) = request.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 1024 * 256).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid body"})),
            )
                .into_response();
        }
    };
    let session_cache_scope = extract_session_cache_scope(&parts.headers, &body_bytes);

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
    let exact_key = build_exact_cache_key("native", &req.prompt, session_cache_scope.as_deref());
    state.exact_cache.put(exact_key, cached_json.clone());

    // L1b semantic cache.
    let embedder = state.text_embedder.clone();
    let prompt_for_embed = namespaced_semantic_cache_input("native", &req.prompt);
    if let Ok(Ok(embedding)) =
        tokio::task::spawn_blocking(move || embedder.generate_embedding(&prompt_for_embed)).await
    {
        state
            .vector_cache
            .insert(embedding, cached_json, session_cache_scope)
            .await;
    }

    tracing::info!(
        prompt_len = req.prompt.len(),
        "Cache store: written to L1a+L1b"
    );

    (StatusCode::OK, Json(serde_json::json!({"stored": true}))).into_response()
}

#[derive(Clone)]
struct InProcessMcpToolExecutor {
    state: Arc<AppState>,
    user_agent: Option<String>,
}

#[async_trait]
impl ToolExecutor for InProcessMcpToolExecutor {
    async fn cache_lookup(&self, prompt: &str) -> anyhow::Result<Option<String>> {
        let request = self.internal_request(
            "/api/v1/cache/lookup",
            serde_json::json!({ "prompt": prompt }),
        )?;
        let response = cache_lookup_handler(request).await.into_response();

        if response.status() == StatusCode::NO_CONTENT {
            return Ok(None);
        }

        if !response.status().is_success() {
            anyhow::bail!("cache lookup failed: {}", response.status());
        }

        let body = axum::body::to_bytes(response.into_body(), 1024 * 128).await?;
        let payload: Value = serde_json::from_slice(&body).unwrap_or(serde_json::json!({}));
        let answer = payload
            .get("message")
            .and_then(|message| message.as_str())
            .or_else(|| {
                payload
                    .get("response")
                    .and_then(|response| response.as_str())
            })
            .unwrap_or("")
            .to_string();

        if answer.is_empty() {
            Ok(None)
        } else {
            Ok(Some(answer))
        }
    }

    async fn cache_store(&self, prompt: &str, response: &str, model: &str) -> anyhow::Result<()> {
        let request = self.internal_request(
            "/api/v1/cache/store",
            serde_json::json!({
                "prompt": prompt,
                "response": response,
                "model": model,
            }),
        )?;
        let response = cache_store_handler(request).await.into_response();
        if !response.status().is_success() {
            anyhow::bail!("cache store failed: {}", response.status());
        }
        Ok(())
    }
}

impl InProcessMcpToolExecutor {
    fn internal_request(&self, uri: &str, body: Value) -> anyhow::Result<Request> {
        let mut builder = Request::builder()
            .method("POST")
            .uri(uri)
            .header(CONTENT_TYPE, "application/json");
        if let Some(user_agent) = self.user_agent.as_deref() {
            builder = builder.header(USER_AGENT, user_agent);
        }
        let mut request = builder.body(Body::from(serde_json::to_vec(&body)?))?;
        request.extensions_mut().insert(self.state.clone());
        Ok(request)
    }
}

/// MCP Streamable HTTP GET endpoint.
pub async fn mcp_http_get_handler(headers: HeaderMap) -> Response {
    let Some(session_id) = extract_mcp_session_id(&headers) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Mcp-Session-Id header is required"})),
        )
            .into_response();
    };

    if !mcp::http_session_exists(&session_id) {
        return StatusCode::NOT_FOUND.into_response();
    }

    let stream = IntervalStream::new(tokio::time::interval(Duration::from_secs(15)))
        .map(|_| Ok::<Event, Infallible>(Event::default().comment("keepalive")));

    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response();
    if let Ok(header_value) = HeaderValue::from_str(&session_id) {
        response
            .headers_mut()
            .insert(mcp::SESSION_HEADER, header_value);
    }
    response
}

/// MCP Streamable HTTP POST endpoint.
pub async fn mcp_http_post_handler(request: Request) -> Response {
    let state = request
        .extensions()
        .get::<Arc<AppState>>()
        .cloned()
        .expect("AppState missing");
    let headers = request.headers().clone();
    let wants_sse = accepts_mcp_sse(&headers);
    let user_agent = headers
        .get(USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);

    let (_parts, body) = request.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 1024 * 256).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid body"})),
            )
                .into_response();
        }
    };

    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(payload) => payload,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid JSON-RPC body"})),
            )
                .into_response();
        }
    };

    let messages: Vec<Value> = match &payload {
        Value::Array(values) => values.clone(),
        _ => vec![payload.clone()],
    };

    if !messages.iter().any(mcp::is_request_message) {
        return StatusCode::ACCEPTED.into_response();
    }

    let initialize_requested = messages
        .iter()
        .any(|message| mcp::message_method(message) == Some("initialize"));
    let session_id = if initialize_requested {
        Some(mcp::register_http_session())
    } else {
        match extract_mcp_session_id(&headers) {
            Some(session_id) if mcp::http_session_exists(&session_id) => Some(session_id),
            Some(_) => return StatusCode::NOT_FOUND.into_response(),
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "Mcp-Session-Id header is required after initialize"})),
                )
                    .into_response();
            }
        }
    };

    let executor = InProcessMcpToolExecutor { state, user_agent };
    let mut responses = Vec::new();
    for message in &messages {
        if let Some(response) =
            mcp::handle_message(message, mcp::STREAMABLE_HTTP_PROTOCOL_VERSION, &executor).await
        {
            responses.push(response);
        }
    }

    if responses.is_empty() {
        return StatusCode::ACCEPTED.into_response();
    }

    let response_payload = if payload.is_array() {
        Value::Array(responses)
    } else {
        responses
            .into_iter()
            .next()
            .unwrap_or_else(|| serde_json::json!({}))
    };

    build_mcp_response(
        response_payload,
        wants_sse,
        session_id.as_deref().filter(|_| initialize_requested),
    )
}

/// MCP Streamable HTTP DELETE endpoint.
pub async fn mcp_http_delete_handler(headers: HeaderMap) -> Response {
    let Some(session_id) = extract_mcp_session_id(&headers) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Mcp-Session-Id header is required"})),
        )
            .into_response();
    };

    if mcp::remove_http_session(&session_id) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

fn accepts_mcp_sse(headers: &HeaderMap) -> bool {
    headers
        .get(ACCEPT)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.contains("text/event-stream"))
        .unwrap_or(false)
}

fn extract_mcp_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get(mcp::SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn build_mcp_response(payload: Value, wants_sse: bool, session_id: Option<&str>) -> Response {
    if wants_sse {
        let body = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
        let stream = tokio_stream::iter(vec![Ok::<Event, Infallible>(
            Event::default().event("message").data(body),
        )]);
        let mut response = Sse::new(stream).into_response();
        if let Some(session_id) = session_id
            && let Ok(header_value) = HeaderValue::from_str(session_id)
        {
            response
                .headers_mut()
                .insert(mcp::SESSION_HEADER, header_value);
        }
        response
    } else {
        let mut response = Json(payload).into_response();
        if let Some(session_id) = session_id
            && let Ok(header_value) = HeaderValue::from_str(session_id)
        {
            response
                .headers_mut()
                .insert(mcp::SESSION_HEADER, header_value);
        }
        response
    }
}

fn record_cache_lookup_prompt(
    final_layer: &str,
    deflected: bool,
    prompt: &str,
    elapsed: std::time::Duration,
    status_code: StatusCode,
    tool: &str,
) {
    match final_layer {
        "l1a" => {
            crate::metrics::record_cache_event_with_tool("l1a", "hit", tool);
            crate::visibility::record_agent_cache_event(tool, "l1a", "hit");
        }
        "l1b" => {
            crate::metrics::record_cache_event_with_tool("l1a", "miss", tool);
            crate::metrics::record_cache_event_with_tool("l1b", "hit", tool);
            crate::visibility::record_agent_cache_event(tool, "l1a", "miss");
            crate::visibility::record_agent_cache_event(tool, "l1b", "hit");
        }
        "miss" => {
            crate::metrics::record_cache_event_with_tool("l1a", "miss", tool);
            crate::metrics::record_cache_event_with_tool("l1b", "miss", tool);
            crate::visibility::record_agent_cache_event(tool, "l1a", "miss");
            crate::visibility::record_agent_cache_event(tool, "l1b", "miss");
        }
        _ => {}
    }
    crate::metrics::record_cache_event_with_tool(
        "l1",
        if deflected { "hit" } else { "miss" },
        tool,
    );
    crate::visibility::record_agent_cache_event(tool, "l1", if deflected { "hit" } else { "miss" });
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

    use crate::config::{AppConfig, FallbackProviderConfig, ProviderQuotaConfig, QuotaLimitAction};
    use crate::core::context_compress::InstructionCache;
    use crate::core::usage::UsageTracker;
    use crate::layer1::embeddings::shared_test_embedder;
    use crate::middleware::body_buffer::buffer_body_middleware;
    use crate::state::AppLlmAgent;
    use crate::visibility::{agent_stats_snapshot, clear_prompt_stats, prompt_stats_snapshot};

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
        let config = Arc::new(AppConfig::test_default());
        AppState::test_with_agent(agent, config)
    }

    fn handler_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/api/chat", post(chat_handler))
            .route(
                "/v1/chat/completions",
                post(openai_chat_completions_handler),
            )
            .route(
                "/v1beta/models/{*rest}",
                post(gemini_generate_content_handler),
            )
            .route("/v1/models", get(openai_models_handler))
            .route("/debug/providers", get(provider_status_handler))
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

    fn mcp_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route(
                "/mcp",
                get(mcp_http_get_handler)
                    .post(mcp_http_post_handler)
                    .delete(mcp_http_delete_handler),
            )
            .route(
                "/mcp/",
                get(mcp_http_get_handler)
                    .post(mcp_http_post_handler)
                    .delete(mcp_http_delete_handler),
            )
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
        let mut config = (*state.config).clone();
        config
            .model_aliases
            .insert("ignored-by-isartor".into(), "gpt-4o-mini".into());
        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
            config: Arc::new(config),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });
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
    async fn gemini_generate_content_returns_gemini_shape() {
        let state = test_state(Arc::new(SuccessAgent));
        let mut config = (*state.config).clone();
        config.external_llm_model = "gemini-2.0-flash".into();
        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
            config: Arc::new(config),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "hello gemini"}]}]
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1beta/models/gemini-2.0-flash:generateContent")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(
            json["candidates"][0]["content"]["parts"][0]["text"],
            "Reply to: user: hello gemini"
        );
        assert_eq!(json["modelVersion"], "gemini-2.0-flash");
    }

    #[tokio::test]
    async fn gemini_stream_generate_content_returns_sse() {
        let state = test_state(Arc::new(SuccessAgent));
        let mut config = (*state.config).clone();
        config.external_llm_model = "gemini-2.0-flash".into();
        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
            config: Arc::new(config),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "hello gemini"}]}]
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(text.contains("\"candidates\""));
        assert!(text.contains("\"text\":\"Reply to: user: hello gemini\""));
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
        config
            .model_aliases
            .insert("client-specified-model".into(), "gpt-4o-mini".into());

        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
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
    async fn openai_tool_request_passthrough_accepts_array_content_parts() {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{body_partial_json, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " from OpenClaw"}
                    ]
                }],
                "tools": [{
                    "type": "function",
                    "function": {"name": "lookup_weather"}
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
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
        config
            .model_aliases
            .insert("client-specified-model".into(), "gpt-4o-mini".into());

        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
            config: Arc::new(config),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });
        let app = handler_app(state);

        let body = serde_json::to_vec(&serde_json::json!({
            "model": "client-specified-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " from OpenClaw"}
                ]
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {"type": "object"}
                }
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
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "lookup_weather"
        );
    }

    #[test]
    fn groq_passthrough_uses_groq_default_when_config_still_points_to_openai() {
        let state = test_state(Arc::new(SuccessAgent));
        let mut config = (*state.config).clone();
        config.llm_provider = LlmProvider::Groq;
        config.external_llm_url = DEFAULT_OPENAI_CHAT_COMPLETIONS_URL.into();

        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
            config: Arc::new(config),
            #[cfg(feature = "embedded-inference")]
            embedded_classifier: None,
        });

        assert_eq!(
            provider_chat_completions_url(state.primary_provider()).as_deref(),
            Some("https://api.groq.com/openai/v1/chat/completions")
        );
    }

    #[tokio::test]
    async fn retryable_primary_failure_falls_back_to_next_provider() {
        use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

        let primary = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limit"))
            .mount(&primary)
            .await;

        let fallback = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_fallback",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "meta/llama-3.1-8b-instruct"
                ,
                "output": [{
                    "type": "message",
                    "id": "msg_fallback",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "Recovered from fallback"
                    }]
                }],
                "tools": []
            })))
            .mount(&fallback)
            .await;

        let mut config = (*test_state(Arc::new(SuccessAgent)).config).clone();
        config.llm_provider = LlmProvider::Cerebras;
        config.external_llm_model = "llama-3.3-70b".into();
        config.external_llm_api_key = "primary-key".into();
        config.external_llm_url = format!("{}/v1/chat/completions", primary.uri());
        config.fallback_providers = vec![FallbackProviderConfig {
            provider: LlmProvider::Nvidia,
            model: "meta/llama-3.1-8b-instruct".into(),
            api_key: "fallback-key".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: crate::config::KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            url: format!("{}/v1/chat/completions", fallback.uri()),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
        }];

        let app = handler_app(Arc::new(AppState::new(
            Arc::new(config),
            shared_test_embedder(),
        )));

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": "hello" })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let provider_header = resp
            .headers()
            .get("x-isartor-provider")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(status, StatusCode::OK, "unexpected response: {json}");
        assert_eq!(provider_header.as_deref(), Some("nvidia"));
        assert_eq!(json["message"], "Recovered from fallback");
        assert_eq!(json["model"], "meta/llama-3.1-8b-instruct");
    }

    #[tokio::test]
    async fn quota_fallback_skips_primary_and_uses_next_provider() {
        use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

        let primary = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("should not be used"))
            .expect(0)
            .mount(&primary)
            .await;

        let fallback = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_quota_fallback",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "meta/llama-3.1-8b-instruct",
                "output": [{
                    "type": "message",
                    "id": "msg_quota_fallback",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "Recovered from quota fallback"
                    }]
                }],
                "tools": []
            })))
            .mount(&fallback)
            .await;

        let mut config = (*test_state(Arc::new(SuccessAgent)).config).clone();
        config.llm_provider = LlmProvider::Cerebras;
        config.external_llm_model = "llama-3.3-70b".into();
        config.external_llm_api_key = "primary-key".into();
        config.external_llm_url = format!("{}/v1/chat/completions", primary.uri());
        config.quota.insert(
            "cerebras".into(),
            ProviderQuotaConfig {
                daily_token_limit: Some(1),
                action_on_limit: QuotaLimitAction::Fallback,
                ..ProviderQuotaConfig::default()
            },
        );
        config.fallback_providers = vec![FallbackProviderConfig {
            provider: LlmProvider::Nvidia,
            model: "meta/llama-3.1-8b-instruct".into(),
            api_key: "fallback-key".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: crate::config::KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            url: format!("{}/v1/chat/completions", fallback.uri()),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
        }];

        let app = handler_app(Arc::new(AppState::new(
            Arc::new(config),
            shared_test_embedder(),
        )));

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": "hello" })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let provider_header = resp
            .headers()
            .get("x-isartor-provider")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(provider_header.as_deref(), Some("nvidia"));
        assert_eq!(json["message"], "Recovered from quota fallback");
    }

    #[tokio::test]
    async fn quota_block_returns_429_without_calling_fallback() {
        use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

        let primary = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("should not be used"))
            .expect(0)
            .mount(&primary)
            .await;

        let fallback = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("should not be used"))
            .expect(0)
            .mount(&fallback)
            .await;

        let mut config = (*test_state(Arc::new(SuccessAgent)).config).clone();
        config.llm_provider = LlmProvider::Cerebras;
        config.external_llm_model = "llama-3.3-70b".into();
        config.external_llm_api_key = "primary-key".into();
        config.external_llm_url = format!("{}/v1/chat/completions", primary.uri());
        config.quota.insert(
            "cerebras".into(),
            ProviderQuotaConfig {
                daily_token_limit: Some(1),
                action_on_limit: QuotaLimitAction::Block,
                ..ProviderQuotaConfig::default()
            },
        );
        config.fallback_providers = vec![FallbackProviderConfig {
            provider: LlmProvider::Nvidia,
            model: "meta/llama-3.1-8b-instruct".into(),
            api_key: "fallback-key".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: crate::config::KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            url: format!("{}/v1/chat/completions", fallback.uri()),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
        }];

        let app = handler_app(Arc::new(AppState::new(
            Arc::new(config),
            shared_test_embedder(),
        )));

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": "hello" })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(json["message"].as_str().unwrap().contains("quota exceeded"));
    }

    #[tokio::test]
    async fn bad_request_does_not_fall_back_to_next_provider() {
        use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

        let primary = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_string("invalid request"))
            .mount(&primary)
            .await;

        let fallback = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_unused",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "meta/llama-3.1-8b-instruct",
                "output": [{
                    "type": "message",
                    "id": "msg_unused",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "should not run"
                    }]
                }],
                "tools": []
            })))
            .expect(0)
            .mount(&fallback)
            .await;

        let mut config = (*test_state(Arc::new(SuccessAgent)).config).clone();
        config.llm_provider = LlmProvider::Cerebras;
        config.external_llm_model = "llama-3.3-70b".into();
        config.external_llm_api_key = "primary-key".into();
        config.external_llm_url = format!("{}/v1/chat/completions", primary.uri());
        config.fallback_providers = vec![FallbackProviderConfig {
            provider: LlmProvider::Nvidia,
            model: "meta/llama-3.1-8b-instruct".into(),
            api_key: "fallback-key".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: crate::config::KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            url: format!("{}/v1/chat/completions", fallback.uri()),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
        }];

        let app = handler_app(Arc::new(AppState::new(
            Arc::new(config),
            shared_test_embedder(),
        )));

        let body = serde_json::to_vec(&serde_json::json!({ "prompt": "hello" })).unwrap();
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
        assert!(
            json["message"]
                .as_str()
                .unwrap()
                .contains("invalid request")
        );
    }

    #[tokio::test]
    async fn provider_status_reports_primary_and_fallback_entries() {
        let mut config = (*test_state(Arc::new(SuccessAgent)).config).clone();
        config.llm_provider = LlmProvider::Cerebras;
        config.external_llm_model = "llama-3.3-70b".into();
        config.external_llm_api_key = "primary-key".into();
        config.external_llm_url = "https://api.cerebras.ai/v1/chat/completions".into();
        config.fallback_providers = vec![FallbackProviderConfig {
            provider: LlmProvider::Nvidia,
            model: "meta/llama-3.1-8b-instruct".into(),
            api_key: "fallback-key".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: crate::config::KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            url: "https://integrate.api.nvidia.com/v1/chat/completions".into(),
            azure_deployment_id: String::new(),
            azure_api_version: "2024-08-01-preview".into(),
        }];

        let app = handler_app(Arc::new(AppState::new(
            Arc::new(config),
            shared_test_embedder(),
        )));

        let req = Request::builder()
            .method("GET")
            .uri("/debug/providers")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["active_provider"], "cerebras");
        assert_eq!(json["providers"].as_array().unwrap().len(), 2);
        assert_eq!(json["providers"][0]["name"], "cerebras");
        assert_eq!(json["providers"][0]["active"], true);
        assert_eq!(json["providers"][1]["name"], "nvidia");
        assert_eq!(json["providers"][1]["active"], false);
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
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
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
    async fn openai_models_include_aliases_and_resolved_targets() {
        let state = test_state(Arc::new(SuccessAgent));
        let mut config = (*state.config).clone();
        config
            .model_aliases
            .insert("fast".into(), "gpt-4o-mini".into());
        config.model_aliases.insert("smart".into(), "gpt-4o".into());

        let state = Arc::new(AppState {
            http_client: reqwest::Client::new(),
            exact_cache: state.exact_cache.clone(),
            vector_cache: state.vector_cache.clone(),
            provider_chain: Arc::new(crate::state::resolved_provider_chain(&config)),
            usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
            llm_agent: Arc::new(SuccessAgent),
            slm_client: state.slm_client.clone(),
            text_embedder: state.text_embedder.clone(),
            instruction_cache: Arc::new(InstructionCache::new()),
            provider_health: Arc::new(crate::state::ProviderHealthTracker::from_config(&config)),
            provider_key_pools: Arc::new(
                crate::state::ProviderKeyPoolManager::from_provider_chain(
                    crate::state::resolved_provider_chain(&config).as_slice(),
                ),
            ),
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
        let ids = json["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();

        assert!(ids.contains(&"gpt-4o-mini".to_string()));
        assert!(ids.contains(&"gpt-4o".to_string()));
        assert!(ids.contains(&"fast".to_string()));
        assert!(ids.contains(&"smart".to_string()));
    }

    #[tokio::test]
    async fn stale_cache_fallback_on_llm_failure() {
        let state = test_state(Arc::new(FailAgent));

        // Pre-populate the exact cache with a stale entry for "fallback test".
        let prompt = "fallback test";
        let key_input = format!("native|{prompt}\nmodel: gpt-4o-mini");
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
    async fn debug_providers_reports_unknown_before_l3_traffic() {
        let state = test_state(Arc::new(SuccessAgent));
        let app = handler_app(state);

        let req = Request::builder()
            .method("GET")
            .uri("/debug/providers")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let snapshot: crate::models::ProviderStatusResponse =
            serde_json::from_slice(&body).unwrap();
        assert_eq!(snapshot.active_provider, "openai");
        assert_eq!(snapshot.providers.len(), 1);
        let entry = &snapshot.providers[0];
        assert_eq!(entry.status, crate::models::ProviderHealthStatus::Unknown);
        assert_eq!(entry.requests_total, 0);
        assert_eq!(entry.errors_total, 0);
    }

    #[tokio::test]
    async fn debug_providers_tracks_successful_l3_requests() {
        let state = test_state(Arc::new(SuccessAgent));
        let app = handler_app(state);

        let chat_req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({ "prompt": "hello" })).unwrap(),
            ))
            .unwrap();
        let chat_resp = app.clone().oneshot(chat_req).await.unwrap();
        assert_eq!(chat_resp.status(), StatusCode::OK);

        let status_req = Request::builder()
            .method("GET")
            .uri("/debug/providers")
            .body(Body::empty())
            .unwrap();
        let status_resp = app.oneshot(status_req).await.unwrap();
        let body = status_resp.into_body().collect().await.unwrap().to_bytes();
        let snapshot: crate::models::ProviderStatusResponse =
            serde_json::from_slice(&body).unwrap();
        let entry = &snapshot.providers[0];
        assert_eq!(entry.status, crate::models::ProviderHealthStatus::Healthy);
        assert_eq!(entry.requests_total, 1);
        assert_eq!(entry.errors_total, 0);
        assert!(entry.last_success.is_some());
    }

    #[tokio::test]
    async fn debug_providers_tracks_failed_l3_requests() {
        let state = test_state(Arc::new(FailAgent));
        let app = handler_app(state);

        let chat_req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({ "prompt": "hello" })).unwrap(),
            ))
            .unwrap();
        let chat_resp = app.clone().oneshot(chat_req).await.unwrap();
        assert_eq!(chat_resp.status(), StatusCode::BAD_GATEWAY);

        let status_req = Request::builder()
            .method("GET")
            .uri("/debug/providers")
            .body(Body::empty())
            .unwrap();
        let status_resp = app.oneshot(status_req).await.unwrap();
        let body = status_resp.into_body().collect().await.unwrap().to_bytes();
        let snapshot: crate::models::ProviderStatusResponse =
            serde_json::from_slice(&body).unwrap();
        let entry = &snapshot.providers[0];
        assert_eq!(entry.status, crate::models::ProviderHealthStatus::Failing);
        assert_eq!(entry.requests_total, 1);
        assert_eq!(entry.errors_total, 1);
        assert!(entry.last_error.is_some());
        assert!(
            entry
                .last_error_message
                .as_deref()
                .unwrap_or_default()
                .contains("provider outage")
        );
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
        let agents = agent_stats_snapshot();
        assert!(
            agents
                .agents
                .values()
                .any(|entry| entry.cache_hits >= 1 && entry.l1a_hits >= 1)
        );
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
        let agents = agent_stats_snapshot();
        assert!(agents.agents.values().any(|entry| {
            entry.cache_misses >= 1 && entry.l1a_misses >= 1 && entry.l1b_misses >= 1
        }));
    }

    #[tokio::test]
    async fn mcp_http_initialize_get_and_delete_manage_sessions() {
        let state = test_state(Arc::new(SuccessAgent));
        let app = mcp_app(state);

        let initialize_req = Request::builder()
            .method("POST")
            .uri("/mcp/")
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {}
                    }
                }))
                .unwrap(),
            ))
            .unwrap();
        let initialize_resp = app.clone().oneshot(initialize_req).await.unwrap();
        assert_eq!(initialize_resp.status(), StatusCode::OK);
        let session_id = initialize_resp
            .headers()
            .get(mcp::SESSION_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap()
            .to_string();

        let initialize_body = initialize_resp
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let initialize_json: Value = serde_json::from_slice(&initialize_body).unwrap();
        assert_eq!(
            initialize_json["result"]["protocolVersion"],
            mcp::STREAMABLE_HTTP_PROTOCOL_VERSION
        );

        let stream_req = Request::builder()
            .method("GET")
            .uri("/mcp/")
            .header("accept", "text/event-stream")
            .header(mcp::SESSION_HEADER, &session_id)
            .body(Body::empty())
            .unwrap();
        let stream_resp = app.clone().oneshot(stream_req).await.unwrap();
        assert_eq!(stream_resp.status(), StatusCode::OK);
        assert_eq!(
            stream_resp
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .unwrap(),
            "text/event-stream"
        );

        let delete_req = Request::builder()
            .method("DELETE")
            .uri("/mcp/")
            .header(mcp::SESSION_HEADER, &session_id)
            .body(Body::empty())
            .unwrap();
        let delete_resp = app.clone().oneshot(delete_req).await.unwrap();
        assert_eq!(delete_resp.status(), StatusCode::NO_CONTENT);

        let missing_stream_req = Request::builder()
            .method("GET")
            .uri("/mcp/")
            .header("accept", "text/event-stream")
            .header(mcp::SESSION_HEADER, &session_id)
            .body(Body::empty())
            .unwrap();
        let missing_stream_resp = app.oneshot(missing_stream_req).await.unwrap();
        assert_eq!(missing_stream_resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn mcp_http_tool_flow_supports_sse_post_responses() {
        clear_prompt_stats();
        let state = test_state(Arc::new(SuccessAgent));
        let app = mcp_app(state);

        let initialize_req = Request::builder()
            .method("POST")
            .uri("/mcp/")
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .header(USER_AGENT, "Cursor/1.0")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {}
                    }
                }))
                .unwrap(),
            ))
            .unwrap();
        let initialize_resp = app.clone().oneshot(initialize_req).await.unwrap();
        let session_id = initialize_resp
            .headers()
            .get(mcp::SESSION_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap()
            .to_string();

        let store_req = Request::builder()
            .method("POST")
            .uri("/mcp/")
            .header("accept", "application/json, text/event-stream")
            .header("content-type", "application/json")
            .header(USER_AGENT, "Cursor/1.0")
            .header(mcp::SESSION_HEADER, &session_id)
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "isartor_cache_store",
                        "arguments": {
                            "prompt": "capital of France",
                            "response": "Paris."
                        }
                    }
                }))
                .unwrap(),
            ))
            .unwrap();
        let store_resp = app.clone().oneshot(store_req).await.unwrap();
        assert_eq!(store_resp.status(), StatusCode::OK);
        assert_eq!(
            store_resp
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .unwrap(),
            "text/event-stream"
        );
        let store_body = store_resp.into_body().collect().await.unwrap().to_bytes();
        let store_json = parse_sse_message(&store_body);
        assert_eq!(
            store_json["result"]["content"][0]["text"],
            "Cached successfully"
        );

        let chat_req = Request::builder()
            .method("POST")
            .uri("/mcp/")
            .header("accept", "application/json, text/event-stream")
            .header("content-type", "application/json")
            .header(USER_AGENT, "Cursor/1.0")
            .header(mcp::SESSION_HEADER, &session_id)
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "isartor_chat",
                        "arguments": {
                            "prompt": "capital of France"
                        }
                    }
                }))
                .unwrap(),
            ))
            .unwrap();
        let chat_resp = app.oneshot(chat_req).await.unwrap();
        assert_eq!(chat_resp.status(), StatusCode::OK);
        let chat_body = chat_resp.into_body().collect().await.unwrap().to_bytes();
        let chat_json = parse_sse_message(&chat_body);
        assert_eq!(chat_json["result"]["content"][0]["text"], "Paris.");

        let stats = prompt_stats_snapshot(50);
        assert!(stats.by_layer.get("l1a").copied().unwrap_or(0) >= 1);
        let agents = agent_stats_snapshot();
        assert!(
            agents
                .agents
                .values()
                .any(|entry| entry.cache_hits >= 1 && entry.l1a_hits >= 1)
        );
    }

    fn parse_sse_message(body: &[u8]) -> Value {
        let text = String::from_utf8_lossy(body);
        let payload = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>()
            .join("\n");
        serde_json::from_str(&payload).unwrap()
    }
}

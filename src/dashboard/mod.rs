//! Embedded web management dashboard.
//!
//! Serves a single-page application at `/dashboard` and exposes a set of
//! authenticated JSON endpoints under `/api/admin/` that the frontend
//! consumes.  All admin API routes reuse the same `AppState` and API-key
//! auth middleware as the rest of the gateway.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::Request;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{
    Json, Router,
    routing::{get, post},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table};

use crate::config::{AppConfig, KeyRotationStrategy, LlmProvider};
use crate::core::quota::current_quota_status_lines;
use crate::core::usage::ProviderModelBreakdown;
use crate::models::{
    ProviderHealthStatus, ProviderStatusEntry, ProviderStatusResponse, UsageStatsResponse,
};
use crate::state::{
    AppState, ProviderHealthStateSnapshot, ResolvedProviderConfig, resolved_provider_chain,
};

// Embedded frontend — compiled into the binary at build time.
const DASHBOARD_HTML: &str = include_str!("index.html");
const LOGO_PNG: &[u8] = include_bytes!("logo.png");
const DEFAULT_AZURE_API_VERSION: &str = "2024-08-01-preview";

// ── Static assets ─────────────────────────────────────────────────────

/// Serve the dashboard SPA (no auth — the shell is static HTML).
pub async fn dashboard_index() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        DASHBOARD_HTML,
    )
}

/// Serve the Isartor logo (no auth).
pub async fn dashboard_logo() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/png")],
        LOGO_PNG,
    )
}

// ── Overview endpoint ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    pub version: &'static str,
    pub provider: String,
    pub model: String,
    pub uptime_secs: u64,
    pub total_requests: u64,
    pub total_deflected: u64,
    pub deflection_rate: f64,
    pub total_tokens: u64,
    pub estimated_cost_usd: f64,
    pub estimated_saved_cost_usd: f64,
    pub l1a_entries: u64,
    pub l1b_entries: u64,
    pub request_logging_enabled: bool,
    pub slm_router_enabled: bool,
    pub offline_mode: bool,
    pub quota_warnings: Vec<String>,
}

pub async fn admin_overview_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };

    let usage = state.usage_tracker.snapshot(Some(168)); // last 7 days
    let total_req = usage.total_requests + usage.total_deflected_requests;
    let primary = state.primary_provider();
    let uptime_secs = state.started_at.elapsed().as_secs();
    let l1b_entries = state.vector_cache.len().await as u64;
    let l1a_entries = state.exact_cache.len() as u64;

    // Collect quota warnings for the primary provider.
    let quota_warnings = current_quota_status_lines(
        &state.config,
        &state.usage_tracker,
        primary.provider_name(),
        Utc::now(),
    )
    .map(|(_, lines)| lines)
    .unwrap_or_default();

    let overview = OverviewResponse {
        version: env!("CARGO_PKG_VERSION"),
        provider: primary.provider_name().to_string(),
        model: primary.configured_model_id().to_string(),
        uptime_secs,
        total_requests: total_req,
        total_deflected: usage.total_deflected_requests,
        deflection_rate: usage.deflection_rate,
        total_tokens: usage.total_tokens,
        estimated_cost_usd: usage.estimated_cost_usd,
        estimated_saved_cost_usd: usage.estimated_saved_cost_usd,
        l1a_entries,
        l1b_entries,
        request_logging_enabled: state.config.enable_request_logs,
        slm_router_enabled: state.config.enable_slm_router,
        offline_mode: state.config.offline_mode,
        quota_warnings,
    };

    (StatusCode::OK, Json(overview)).into_response()
}

// ── Providers endpoint ────────────────────────────────────────────────

fn provider_runtime_key(name: &str, raw_model: Option<&str>, active: bool) -> String {
    format!(
        "{}::{}::{}",
        name,
        raw_model.unwrap_or_default(),
        if active { "active" } else { "fallback" }
    )
}

fn provider_config_key(provider: &ResolvedProviderConfig) -> String {
    provider_runtime_key(
        provider.provider_name(),
        Some(&provider.model),
        provider.active,
    )
}

fn provider_health_key(provider: &ResolvedProviderConfig) -> String {
    format!(
        "{}::{}",
        provider.provider_name(),
        provider.configured_model_id()
    )
}

fn key_rotation_strategy_label(strategy: &KeyRotationStrategy) -> String {
    match strategy {
        KeyRotationStrategy::RoundRobin => "round_robin".to_string(),
        KeyRotationStrategy::Priority => "priority".to_string(),
    }
}

fn build_dashboard_provider_response(
    cfg: &AppConfig,
    runtime: ProviderStatusResponse,
    health_states: &std::collections::HashMap<String, ProviderHealthStateSnapshot>,
) -> ProviderStatusResponse {
    let runtime_by_key: std::collections::HashMap<String, ProviderStatusEntry> = runtime
        .providers
        .into_iter()
        .map(|entry| {
            (
                provider_runtime_key(&entry.name, entry.raw_model.as_deref(), entry.active),
                entry,
            )
        })
        .collect();
    let providers = resolved_provider_chain(cfg)
        .into_iter()
        .enumerate()
        .map(|(index, provider)| {
            let active = provider.active;
            let provider_name = provider.provider_name().to_string();
            let raw_model = provider.model.clone();
            let fallback_index = if active { None } else { Some(index - 1) };
            if let Some(mut entry) = runtime_by_key.get(&provider_config_key(&provider)).cloned() {
                entry.config_index = fallback_index;
                entry.raw_model = Some(raw_model);
                entry.config_url = Some(provider.endpoint.clone());
                entry.azure_deployment_id = if provider.provider == LlmProvider::Azure {
                    Some(provider.azure_deployment_id.clone())
                } else {
                    None
                };
                entry.azure_api_version = if provider.provider == LlmProvider::Azure {
                    Some(provider.azure_api_version.clone())
                } else {
                    None
                };
                entry
            } else {
                let health_state = health_states.get(&provider_health_key(&provider));
                ProviderStatusEntry {
                    name: provider_name,
                    active,
                    status: health_state
                        .map(|state| state.status)
                        .unwrap_or(ProviderHealthStatus::Unknown),
                    model: provider.configured_model_id().to_string(),
                    raw_model: Some(raw_model),
                    endpoint: provider.endpoint.clone(),
                    config_url: Some(provider.endpoint.clone()),
                    api_key_configured: matches!(provider.provider, LlmProvider::Ollama)
                        || !provider.provider_keys.is_empty()
                        || !provider.api_key.trim().is_empty(),
                    endpoint_configured: !provider.endpoint.trim().is_empty(),
                    config_index: fallback_index,
                    azure_deployment_id: if provider.provider == LlmProvider::Azure {
                        Some(provider.azure_deployment_id.clone())
                    } else {
                        None
                    },
                    azure_api_version: if provider.provider == LlmProvider::Azure {
                        Some(provider.azure_api_version.clone())
                    } else {
                        None
                    },
                    requests_total: health_state.map(|state| state.requests_total).unwrap_or(0),
                    errors_total: health_state.map(|state| state.errors_total).unwrap_or(0),
                    key_rotation_strategy: key_rotation_strategy_label(
                        &provider.key_rotation_strategy,
                    ),
                    key_cooldown_secs: provider.key_cooldown_secs,
                    keys: Vec::new(),
                    last_success: health_state.and_then(|state| state.last_success.clone()),
                    last_error: health_state.and_then(|state| state.last_error.clone()),
                    last_error_message: health_state
                        .and_then(|state| state.last_error_message.clone()),
                }
            }
        })
        .collect();
    ProviderStatusResponse {
        active_provider: cfg.llm_provider.as_str().to_string(),
        providers,
    }
}

pub async fn admin_providers_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };

    let cfg = load_dashboard_config(&state.config);
    let runtime = state.provider_status();
    let health_states = state.provider_health.health_state_snapshot();
    let resp = build_dashboard_provider_response(&cfg, runtime, &health_states);
    (StatusCode::OK, Json(resp)).into_response()
}

// ── Provider connectivity test endpoint ───────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ProviderTestRequest {
    pub url: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ProviderTestResponse {
    pub reachable: bool,
    pub latency_ms: Option<u64>,
    pub status_code: Option<u16>,
    pub error: Option<String>,
}

fn provider_health_probe_url(url: &str) -> String {
    let without_query = url
        .trim()
        .split('?')
        .next()
        .unwrap_or(url)
        .trim_end_matches('/');
    let base = without_query.trim_end_matches("/chat/completions");
    format!("{}/models", base.trim_end_matches('/'))
}

fn provider_probe_reachable(status: u16) -> bool {
    matches!(status, 200 | 401 | 403 | 404)
}

async fn probe_provider_endpoint(
    http_client: &reqwest::Client,
    target_url: &str,
    api_key: Option<&str>,
) -> ProviderTestResponse {
    let health_url = provider_health_probe_url(target_url);
    let start = Instant::now();
    let mut req_builder = http_client.get(&health_url);
    if let Some(key) = api_key
        && !key.is_empty()
    {
        req_builder = req_builder.bearer_auth(key);
    }

    match req_builder.send().await {
        Ok(resp) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            let status = resp.status().as_u16();
            ProviderTestResponse {
                reachable: provider_probe_reachable(status),
                latency_ms: Some(latency_ms),
                status_code: Some(status),
                error: None,
            }
        }
        Err(e) => ProviderTestResponse {
            reachable: false,
            latency_ms: Some(start.elapsed().as_millis() as u64),
            status_code: None,
            error: Some(e.to_string()),
        },
    }
}

fn matching_provider_by_url(cfg: &AppConfig, target_url: &str) -> Option<ResolvedProviderConfig> {
    let normalized = target_url.trim_end_matches('/');
    resolved_provider_chain(cfg)
        .into_iter()
        .find(|provider| provider.endpoint.trim_end_matches('/') == normalized)
}

fn sync_probe_health(
    state: &AppState,
    provider: &ResolvedProviderConfig,
    response: &ProviderTestResponse,
) {
    if response.reachable {
        state.provider_health.record_probe_success(provider);
    } else {
        let message = response
            .error
            .clone()
            .or_else(|| response.status_code.map(|status| format!("HTTP {status}")));
        state
            .provider_health
            .record_probe_failure(provider, message);
    }
}

pub async fn refresh_provider_health(state: Arc<AppState>) {
    for provider in state.provider_chain.iter() {
        let api_key = provider.provider_keys.first().and_then(|entry| {
            if entry.key.trim().is_empty() {
                None
            } else {
                Some(entry.key.as_str())
            }
        });
        let api_key = api_key.or_else(|| {
            if provider.api_key.trim().is_empty() {
                None
            } else {
                Some(provider.api_key.as_str())
            }
        });
        let response =
            probe_provider_endpoint(&state.http_client, &provider.endpoint, api_key).await;
        sync_probe_health(&state, provider, &response);
    }
}

pub async fn admin_providers_test_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };

    let (parts, body) = request.into_parts();
    let _ = parts; // state already extracted above
    let bytes = match axum::body::to_bytes(body, 8 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("body read error: {e}")})),
            )
                .into_response();
        }
    };
    let req_body: ProviderTestRequest = match serde_json::from_slice(&bytes) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("invalid JSON: {e}")})),
            )
                .into_response();
        }
    };

    let target_url = req_body.url.trim_end_matches('/').to_string();
    let response =
        probe_provider_endpoint(&state.http_client, &target_url, req_body.api_key.as_deref()).await;

    let cfg = load_dashboard_config(&state.config);
    if let Some(provider) = matching_provider_by_url(&cfg, &target_url) {
        sync_probe_health(&state, &provider, &response);
    }

    (StatusCode::OK, Json(response)).into_response()
}

#[derive(Debug, Deserialize)]
pub struct ProviderUpsertRequest {
    pub index: Option<usize>,
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub url: Option<String>,
    pub azure_deployment_id: Option<String>,
    pub azure_api_version: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProviderIndexRequest {
    pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct ProviderMoveRequest {
    pub index: usize,
    pub direction: i32,
}

fn parse_provider_kind(raw: &str) -> Result<LlmProvider, String> {
    serde_json::from_value(serde_json::Value::String(raw.trim().to_lowercase()))
        .map_err(|_| format!("unsupported provider: {}", raw.trim()))
}

fn load_dashboard_doc() -> Result<DocumentMut, String> {
    let raw = std::fs::read_to_string(CONFIG_FILE).unwrap_or_default();
    raw.parse()
        .map_err(|e| format!("cannot parse {CONFIG_FILE}: {e}"))
}

fn write_dashboard_doc(doc: DocumentMut) -> Response {
    match std::fs::write(CONFIG_FILE, doc.to_string()) {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "message": "isartor.toml updated. Restart the gateway to apply changes."
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("write failed: {e}")})),
        )
            .into_response(),
    }
}

fn fallback_provider_tables_mut(doc: &mut DocumentMut) -> &mut ArrayOfTables {
    if doc.get("fallback_providers").is_none() {
        doc["fallback_providers"] = Item::ArrayOfTables(ArrayOfTables::new());
    }
    doc["fallback_providers"]
        .as_array_of_tables_mut()
        .expect("fallback_providers must be an array of tables")
}

fn apply_provider_table(
    table: &mut Table,
    req: &ProviderUpsertRequest,
    provider_kind: &LlmProvider,
    preserve_existing_key: bool,
) -> Result<(), String> {
    let model = req.model.trim();
    if model.is_empty() {
        return Err("provider model cannot be empty".into());
    }

    table["provider"] = toml_edit::value(provider_kind.as_str());
    table["model"] = toml_edit::value(model);

    match req.api_key.as_deref() {
        Some(key) if !key.trim().is_empty() => {
            table["api_key"] = toml_edit::value(key.trim());
        }
        Some(_) if !preserve_existing_key => {
            return Err("api_key cannot be empty when adding a provider".into());
        }
        None if !preserve_existing_key => {
            return Err("api_key is required when adding a provider".into());
        }
        _ => {}
    }

    table["url"] = toml_edit::value(req.url.as_deref().unwrap_or("").trim());

    if *provider_kind == LlmProvider::Azure {
        let deployment = req.azure_deployment_id.as_deref().unwrap_or("").trim();
        if deployment.is_empty() {
            return Err("azure_deployment_id is required when provider is azure".into());
        }
        table["azure_deployment_id"] = toml_edit::value(deployment);
        table["azure_api_version"] = toml_edit::value(
            req.azure_api_version
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(DEFAULT_AZURE_API_VERSION)
                .trim(),
        );
    } else {
        table["azure_deployment_id"] = toml_edit::value("");
        table["azure_api_version"] = toml_edit::value(DEFAULT_AZURE_API_VERSION);
    }

    Ok(())
}

async fn parse_json_body<T: for<'de> Deserialize<'de>>(
    request: Request,
    max_bytes: usize,
) -> Result<T, Response> {
    let (_parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, max_bytes).await {
        Ok(b) => b,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("body read error: {e}")})),
            )
                .into_response());
        }
    };
    serde_json::from_slice(&bytes).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("invalid JSON: {e}")})),
        )
            .into_response()
    })
}

pub async fn admin_providers_add_handler(request: Request) -> Response {
    let req: ProviderUpsertRequest = match parse_json_body(request, 16 * 1024).await {
        Ok(req) => req,
        Err(resp) => return resp,
    };
    let provider_kind = match parse_provider_kind(&req.provider) {
        Ok(provider) => provider,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };
    let mut doc = match load_dashboard_doc() {
        Ok(doc) => doc,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };

    let providers = fallback_provider_tables_mut(&mut doc);
    let mut table = Table::new();
    if let Err(error) = apply_provider_table(&mut table, &req, &provider_kind, false) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "error": error })),
        )
            .into_response();
    }
    providers.push(table);
    write_dashboard_doc(doc)
}

pub async fn admin_providers_edit_handler(request: Request) -> Response {
    let req: ProviderUpsertRequest = match parse_json_body(request, 16 * 1024).await {
        Ok(req) => req,
        Err(resp) => return resp,
    };
    let index = match req.index {
        Some(index) => index,
        None => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": "index is required for provider edits" })),
            )
                .into_response();
        }
    };
    let provider_kind = match parse_provider_kind(&req.provider) {
        Ok(provider) => provider,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };
    let mut doc = match load_dashboard_doc() {
        Ok(doc) => doc,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };

    let providers = fallback_provider_tables_mut(&mut doc);
    let Some(table) = providers.get_mut(index) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "fallback provider not found" })),
        )
            .into_response();
    };
    if let Err(error) = apply_provider_table(table, &req, &provider_kind, true) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "error": error })),
        )
            .into_response();
    }
    write_dashboard_doc(doc)
}

pub async fn admin_providers_remove_handler(request: Request) -> Response {
    let req: ProviderIndexRequest = match parse_json_body(request, 8 * 1024).await {
        Ok(req) => req,
        Err(resp) => return resp,
    };
    let mut doc = match load_dashboard_doc() {
        Ok(doc) => doc,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };
    let providers = fallback_provider_tables_mut(&mut doc);
    if req.index >= providers.len() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "fallback provider not found" })),
        )
            .into_response();
    }
    providers.remove(req.index);
    write_dashboard_doc(doc)
}

pub async fn admin_providers_move_handler(request: Request) -> Response {
    let req: ProviderMoveRequest = match parse_json_body(request, 8 * 1024).await {
        Ok(req) => req,
        Err(resp) => return resp,
    };
    let mut doc = match load_dashboard_doc() {
        Ok(doc) => doc,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error })),
            )
                .into_response();
        }
    };
    let providers = fallback_provider_tables_mut(&mut doc);
    if req.index >= providers.len() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "fallback provider not found" })),
        )
            .into_response();
    }
    let target_index = match req.direction.cmp(&0) {
        std::cmp::Ordering::Less if req.index > 0 => req.index - 1,
        std::cmp::Ordering::Greater if req.index + 1 < providers.len() => req.index + 1,
        _ => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": "provider cannot be moved in that direction" })),
            )
                .into_response();
        }
    };
    let mut reordered: Vec<Table> = providers.iter().cloned().collect();
    let table = reordered.remove(req.index);
    reordered.insert(target_index, table);
    doc["fallback_providers"] = Item::ArrayOfTables(ArrayOfTables::new());
    let providers = fallback_provider_tables_mut(&mut doc);
    for table in reordered {
        providers.push(table);
    }
    write_dashboard_doc(doc)
}

// ── Usage endpoint ────────────────────────────────────────────────────

pub async fn admin_usage_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };

    let window_hours = state.config.usage_window_hours;
    let snapshot: UsageStatsResponse = state.usage_tracker.snapshot(Some(window_hours));
    (StatusCode::OK, Json(snapshot)).into_response()
}

// ── Usage breakdown endpoint ──────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct UsageBreakdownResponse {
    pub window_hours: u64,
    pub rows: Vec<ProviderModelBreakdown>,
    pub quota_status: Vec<ProviderQuotaStatus>,
}

#[derive(Debug, Serialize)]
pub struct ProviderQuotaStatus {
    pub provider: String,
    pub action: String,
    pub lines: Vec<String>,
}

pub async fn admin_usage_breakdown_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };

    let window_hours = state.config.usage_window_hours;
    let rows = state.usage_tracker.snapshot_by_provider(Some(window_hours));

    // Collect quota status for every unique provider in the breakdown.
    let now = Utc::now();
    let providers: Vec<String> = {
        let mut seen = std::collections::BTreeSet::new();
        rows.iter()
            .filter(|r| seen.insert(r.provider.clone()))
            .map(|r| r.provider.clone())
            .collect()
    };

    let quota_status: Vec<ProviderQuotaStatus> = providers
        .iter()
        .filter_map(|p| {
            current_quota_status_lines(&state.config, &state.usage_tracker, p, now).map(
                |(action, lines)| ProviderQuotaStatus {
                    provider: p.clone(),
                    action: format!("{action:?}").to_lowercase(),
                    lines,
                },
            )
        })
        .collect();

    (
        StatusCode::OK,
        Json(UsageBreakdownResponse {
            window_hours,
            rows,
            quota_status,
        }),
    )
        .into_response()
}

// ── Recent requests endpoint ──────────────────────────────────────────

const RECENT_REQUESTS_LIMIT: usize = 100;

#[derive(Debug, Serialize)]
pub struct RecentRequestsResponse {
    pub enabled: bool,
    pub log_path: String,
    pub entries: Vec<serde_json::Value>,
}

pub async fn admin_requests_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };

    if !state.config.enable_request_logs {
        return (
            StatusCode::OK,
            Json(RecentRequestsResponse {
                enabled: false,
                log_path: String::new(),
                entries: vec![],
            }),
        )
            .into_response();
    }

    let log_path = resolve_tilde(&state.config.request_log_path);
    let log_file = std::path::Path::new(&log_path).join("requests.log");

    let entries = read_last_jsonl_lines(&log_file, RECENT_REQUESTS_LIMIT);
    (
        StatusCode::OK,
        Json(RecentRequestsResponse {
            enabled: true,
            log_path: log_file.display().to_string(),
            entries,
        }),
    )
        .into_response()
}

/// Read the last `n` newline-delimited JSON objects from a file.
fn read_last_jsonl_lines(path: &std::path::Path, n: usize) -> Vec<serde_json::Value> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    content
        .lines()
        .rev()
        .take(n)
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn resolve_tilde(path: &str) -> String {
    if let Some(stripped) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{stripped}");
    }
    path.to_string()
}

// ── Config view / edit ────────────────────────────────────────────────

const CONFIG_FILE: &str = "isartor.toml";

/// Safe read-only view of the current config (no secrets exposed).
#[derive(Debug, Serialize)]
pub struct ConfigView {
    pub config_file: String,
    pub file_exists: bool,
    // Gateway
    pub host_port: String,
    pub proxy_port: String,
    pub offline_mode: bool,
    // Provider (L3)
    pub llm_provider: String,
    pub external_llm_model: String,
    pub external_llm_url: String,
    pub azure_deployment_id: String,
    pub azure_api_version: String,
    // L1 cache
    pub cache_mode: String,
    pub cache_ttl_secs: u64,
    pub cache_max_capacity: u64,
    pub similarity_threshold: f64,
    // L2 SLM
    pub enable_slm_router: bool,
    pub local_slm_url: String,
    pub local_slm_model: String,
    pub layer2_sidecar_url: String,
    pub layer2_model_name: String,
    pub layer2_timeout_seconds: u64,
    // Request logging
    pub enable_request_logs: bool,
    pub request_log_path: String,
    pub usage_window_hours: u64,
    pub provider_health_check_interval_secs: u64,
    // Classifier routing
    pub classifier_routing_enabled: bool,
    pub classifier_routing_artifacts_path: String,
    pub classifier_routing_confidence_threshold: f64,
    pub classifier_routing_fallback_to_existing_routing: bool,
    pub classifier_routing_matrix: HashMap<String, HashMap<String, String>>,
    pub classifier_routing_rules: Vec<ClassifierRoutingRuleView>,
}

/// Simplified view of a classifier routing rule for the dashboard.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClassifierRoutingRuleView {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complexity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Fields the user may update via the dashboard (no secrets).
#[derive(Debug, Deserialize)]
pub struct ConfigUpdate {
    pub host_port: Option<String>,
    pub proxy_port: Option<String>,
    pub offline_mode: Option<bool>,
    pub llm_provider: Option<String>,
    pub external_llm_model: Option<String>,
    pub external_llm_url: Option<String>,
    pub azure_deployment_id: Option<String>,
    pub azure_api_version: Option<String>,
    pub cache_mode: Option<String>,
    pub cache_ttl_secs: Option<u64>,
    pub cache_max_capacity: Option<u64>,
    pub similarity_threshold: Option<f64>,
    pub enable_slm_router: Option<bool>,
    pub local_slm_url: Option<String>,
    pub local_slm_model: Option<String>,
    pub layer2_sidecar_url: Option<String>,
    pub layer2_model_name: Option<String>,
    pub layer2_timeout_seconds: Option<u64>,
    pub enable_request_logs: Option<bool>,
    pub request_log_path: Option<String>,
    pub usage_window_hours: Option<u64>,
    pub provider_health_check_interval_secs: Option<u64>,
    // Classifier routing
    pub classifier_routing_enabled: Option<bool>,
    pub classifier_routing_artifacts_path: Option<String>,
    pub classifier_routing_confidence_threshold: Option<f64>,
    pub classifier_routing_fallback_to_existing_routing: Option<bool>,
    pub classifier_routing_matrix: Option<HashMap<String, HashMap<String, String>>>,
    pub classifier_routing_rules: Option<Vec<ClassifierRoutingRuleView>>,
}

fn load_dashboard_config(current: &AppConfig) -> AppConfig {
    if Path::new(CONFIG_FILE).exists() {
        AppConfig::load_with_validation(false).unwrap_or_else(|_| current.clone())
    } else {
        current.clone()
    }
}

fn config_view_from_app_config(cfg: &AppConfig) -> ConfigView {
    ConfigView {
        config_file: CONFIG_FILE.to_string(),
        file_exists: Path::new(CONFIG_FILE).exists(),
        host_port: cfg.host_port.clone(),
        proxy_port: cfg.proxy_port.clone(),
        offline_mode: cfg.offline_mode,
        llm_provider: cfg.llm_provider.as_str().to_string(),
        external_llm_model: cfg.external_llm_model.clone(),
        external_llm_url: cfg.external_llm_url.clone(),
        azure_deployment_id: cfg.azure_deployment_id.clone(),
        azure_api_version: cfg.azure_api_version.clone(),
        cache_mode: format!("{:?}", cfg.cache_mode).to_lowercase(),
        cache_ttl_secs: cfg.cache_ttl_secs,
        cache_max_capacity: cfg.cache_max_capacity,
        similarity_threshold: cfg.similarity_threshold,
        enable_slm_router: cfg.enable_slm_router,
        local_slm_url: cfg.local_slm_url.clone(),
        local_slm_model: cfg.local_slm_model.clone(),
        layer2_sidecar_url: cfg.layer2.sidecar_url.clone(),
        layer2_model_name: cfg.layer2.model_name.clone(),
        layer2_timeout_seconds: cfg.layer2.timeout_seconds,
        enable_request_logs: cfg.enable_request_logs,
        request_log_path: cfg.request_log_path.clone(),
        usage_window_hours: cfg.usage_window_hours,
        provider_health_check_interval_secs: cfg.provider_health_check_interval_secs,
        classifier_routing_enabled: cfg.classifier_routing.enabled,
        classifier_routing_artifacts_path: cfg.classifier_routing.artifacts_path.clone(),
        classifier_routing_confidence_threshold: cfg.classifier_routing.confidence_threshold as f64,
        classifier_routing_fallback_to_existing_routing: cfg
            .classifier_routing
            .fallback_to_existing_routing,
        classifier_routing_matrix: cfg.classifier_routing.matrix.clone(),
        classifier_routing_rules: cfg
            .classifier_routing
            .rules
            .iter()
            .map(|r| ClassifierRoutingRuleView {
                name: r.name.clone(),
                task_type: r.task_type.clone(),
                complexity: r.complexity.clone(),
                persona: r.persona.clone(),
                domain: r.domain.clone(),
                provider: r.provider.clone(),
                model: r.model.clone(),
            })
            .collect(),
    }
}

pub async fn admin_config_get_handler(request: Request) -> impl IntoResponse {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };

    let cfg = load_dashboard_config(&state.config);
    let view = config_view_from_app_config(&cfg);
    (StatusCode::OK, Json(view)).into_response()
}

pub async fn admin_config_post_handler(request: Request) -> impl IntoResponse {
    // Extract state and body.
    let (parts, body) = request.into_parts();
    let state = match parts.extensions.get::<Arc<AppState>>() {
        Some(s) => s.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "missing state"})),
            )
                .into_response();
        }
    };
    let bytes = match axum::body::to_bytes(body, 64 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("body read error: {e}")})),
            )
                .into_response();
        }
    };
    let update: ConfigUpdate = match serde_json::from_slice(&bytes) {
        Ok(u) => u,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("invalid JSON: {e}")})),
            )
                .into_response();
        }
    };

    // Validate a few fields eagerly before touching the file.
    if let Some(ref port) = update.host_port
        && port.trim().is_empty()
    {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": "host_port cannot be empty"})),
        )
            .into_response();
    }
    if let Some(thresh) = update.similarity_threshold
        && !(0.0..=1.0).contains(&thresh)
    {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": "similarity_threshold must be between 0.0 and 1.0"})),
        )
            .into_response();
    }
    let effective_provider = update
        .llm_provider
        .as_deref()
        .unwrap_or(state.config.llm_provider.as_str());
    let effective_azure_deployment_id = update
        .azure_deployment_id
        .as_deref()
        .unwrap_or(&state.config.azure_deployment_id);
    if effective_provider.eq_ignore_ascii_case("azure")
        && effective_azure_deployment_id.trim().is_empty()
    {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": "azure_deployment_id is required when llm_provider is azure"})),
        )
            .into_response();
    }

    // Load or create the TOML document, preserving existing content.
    let raw = std::fs::read_to_string(CONFIG_FILE).unwrap_or_default();
    let mut doc: DocumentMut = match raw.parse() {
        Ok(d) => d,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("cannot parse {CONFIG_FILE}: {e}")})),
            )
                .into_response();
        }
    };

    // Apply updates — top-level keys.
    macro_rules! set_str {
        ($key:expr, $val:expr) => {
            if let Some(v) = $val {
                doc[$key] = toml_edit::value(v);
            }
        };
    }
    macro_rules! set_bool {
        ($key:expr, $val:expr) => {
            if let Some(v) = $val {
                doc[$key] = toml_edit::value(v);
            }
        };
    }
    macro_rules! set_u64 {
        ($key:expr, $val:expr) => {
            if let Some(v) = $val {
                doc[$key] = toml_edit::value(v as i64);
            }
        };
    }
    macro_rules! set_f64 {
        ($key:expr, $val:expr) => {
            if let Some(v) = $val {
                doc[$key] = toml_edit::value(v);
            }
        };
    }

    set_str!("host_port", update.host_port);
    set_str!("proxy_port", update.proxy_port);
    set_bool!("offline_mode", update.offline_mode);
    set_str!("llm_provider", update.llm_provider);
    set_str!("external_llm_model", update.external_llm_model);
    set_str!("external_llm_url", update.external_llm_url);
    set_str!("azure_deployment_id", update.azure_deployment_id);
    set_str!("azure_api_version", update.azure_api_version);
    set_str!("cache_mode", update.cache_mode);
    set_u64!("cache_ttl_secs", update.cache_ttl_secs);
    set_u64!("cache_max_capacity", update.cache_max_capacity);
    set_f64!("similarity_threshold", update.similarity_threshold);
    set_bool!("enable_slm_router", update.enable_slm_router);
    set_str!("local_slm_url", update.local_slm_url);
    set_str!("local_slm_model", update.local_slm_model);
    set_bool!("enable_request_logs", update.enable_request_logs);
    set_str!("request_log_path", update.request_log_path);
    set_u64!("usage_window_hours", update.usage_window_hours);
    set_u64!(
        "provider_health_check_interval_secs",
        update.provider_health_check_interval_secs
    );

    // Apply nested [layer2] keys.
    if update.layer2_sidecar_url.is_some()
        || update.layer2_model_name.is_some()
        || update.layer2_timeout_seconds.is_some()
    {
        if doc.get("layer2").is_none() {
            doc["layer2"] = toml_edit::table();
        }
        if let Some(v) = update.layer2_sidecar_url {
            doc["layer2"]["sidecar_url"] = toml_edit::value(v);
        }
        if let Some(v) = update.layer2_model_name {
            doc["layer2"]["model_name"] = toml_edit::value(v);
        }
        if let Some(v) = update.layer2_timeout_seconds {
            doc["layer2"]["timeout_seconds"] = toml_edit::value(v as i64);
        }
    }

    // Apply nested [classifier_routing] keys.
    if update.classifier_routing_enabled.is_some()
        || update.classifier_routing_artifacts_path.is_some()
        || update.classifier_routing_confidence_threshold.is_some()
        || update
            .classifier_routing_fallback_to_existing_routing
            .is_some()
        || update.classifier_routing_matrix.is_some()
        || update.classifier_routing_rules.is_some()
    {
        if doc.get("classifier_routing").is_none() {
            doc["classifier_routing"] = toml_edit::table();
        }
        if let Some(v) = update.classifier_routing_enabled {
            doc["classifier_routing"]["enabled"] = toml_edit::value(v);
        }
        if let Some(v) = update.classifier_routing_artifacts_path {
            doc["classifier_routing"]["artifacts_path"] = toml_edit::value(v);
        }
        if let Some(v) = update.classifier_routing_confidence_threshold {
            doc["classifier_routing"]["confidence_threshold"] = toml_edit::value(v);
        }
        if let Some(v) = update.classifier_routing_fallback_to_existing_routing {
            doc["classifier_routing"]["fallback_to_existing_routing"] = toml_edit::value(v);
        }
        if let Some(ref matrix) = update.classifier_routing_matrix {
            // Remove existing matrix subtable, then rebuild.
            if let Some(cr) = doc.get_mut("classifier_routing")
                && let Some(tbl) = cr.as_table_mut()
            {
                tbl.remove("matrix");
            }
            if !matrix.is_empty() {
                doc["classifier_routing"]["matrix"] = toml_edit::table();
                for (complexity, task_map) in matrix {
                    doc["classifier_routing"]["matrix"][complexity.as_str()] = toml_edit::table();
                    for (task_type, target) in task_map {
                        doc["classifier_routing"]["matrix"][complexity.as_str()]
                            [task_type.as_str()] = toml_edit::value(target.as_str());
                    }
                }
            }
        }
        if let Some(ref rules) = update.classifier_routing_rules {
            // Remove existing [[classifier_routing.rules]] array, then rebuild.
            if let Some(cr) = doc.get_mut("classifier_routing")
                && let Some(tbl) = cr.as_table_mut()
            {
                tbl.remove("rules");
            }
            if !rules.is_empty() {
                let mut arr = ArrayOfTables::new();
                for rule in rules {
                    let mut t = Table::new();
                    t.insert("name", toml_edit::value(&rule.name));
                    if let Some(ref v) = rule.task_type {
                        t.insert("task_type", toml_edit::value(v.as_str()));
                    }
                    if let Some(ref v) = rule.complexity {
                        t.insert("complexity", toml_edit::value(v.as_str()));
                    }
                    if let Some(ref v) = rule.persona {
                        t.insert("persona", toml_edit::value(v.as_str()));
                    }
                    if let Some(ref v) = rule.domain {
                        t.insert("domain", toml_edit::value(v.as_str()));
                    }
                    if let Some(ref v) = rule.provider {
                        t.insert("provider", toml_edit::value(v.as_str()));
                    }
                    if let Some(ref v) = rule.model {
                        t.insert("model", toml_edit::value(v.as_str()));
                    }
                    arr.push(t);
                }
                doc["classifier_routing"]["rules"] = Item::ArrayOfTables(arr);
            }
        }
    }

    // Live reload is not supported — the caller must restart the gateway.
    let _ = &state.config;

    match std::fs::write(CONFIG_FILE, doc.to_string()) {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "message": "isartor.toml updated. Restart the gateway to apply changes."
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("write failed: {e}")})),
        )
            .into_response(),
    }
}

// ── Router builders ───────────────────────────────────────────────────

/// Unauthenticated router — serves the dashboard shell HTML and static assets.
pub fn dashboard_static_router() -> Router {
    Router::new()
        .route(
            "/dashboard",
            get(|| async { axum::response::Redirect::permanent("/dashboard/") }),
        )
        .route("/dashboard/", get(dashboard_index))
        .route("/dashboard/logo.png", get(dashboard_logo))
}

/// Authenticated router — all endpoints require the gateway API key.
/// Caller is responsible for layering auth middleware.
pub fn admin_api_routes() -> Router {
    Router::new()
        .route("/api/admin/overview", get(admin_overview_handler))
        .route("/api/admin/providers", get(admin_providers_handler))
        .route(
            "/api/admin/providers/add",
            post(admin_providers_add_handler),
        )
        .route(
            "/api/admin/providers/edit",
            post(admin_providers_edit_handler),
        )
        .route(
            "/api/admin/providers/remove",
            post(admin_providers_remove_handler),
        )
        .route(
            "/api/admin/providers/move",
            post(admin_providers_move_handler),
        )
        .route(
            "/api/admin/providers/test",
            post(admin_providers_test_handler),
        )
        .route("/api/admin/usage", get(admin_usage_handler))
        .route(
            "/api/admin/usage/breakdown",
            get(admin_usage_breakdown_handler),
        )
        .route("/api/admin/requests", get(admin_requests_handler))
        .route(
            "/api/admin/config",
            get(admin_config_get_handler).post(admin_config_post_handler),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    use tempfile::tempdir;

    fn cwd_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn dashboard_config_prefers_persisted_file_values() {
        let _guard = cwd_lock().lock().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = tempdir().unwrap();
        std::env::set_current_dir(temp_dir.path()).unwrap();

        std::fs::write(
            temp_dir.path().join(CONFIG_FILE),
            r#"
llm_provider = "openai"
external_llm_model = "gpt-4o-mini"
external_llm_url = "https://api.openai.com/v1"
enable_request_logs = true
request_log_path = "/tmp/isartor-dashboard-logs"
"#,
        )
        .unwrap();

        let runtime_cfg = AppConfig::test_default();
        let loaded = load_dashboard_config(&runtime_cfg);

        std::env::set_current_dir(original_dir).unwrap();

        assert!(loaded.enable_request_logs);
        assert_eq!(loaded.request_log_path, "/tmp/isartor-dashboard-logs");
    }

    #[test]
    fn edit_provider_preserves_existing_api_key_when_left_blank() {
        let mut table = Table::new();
        table["provider"] = toml_edit::value("openai");
        table["model"] = toml_edit::value("gpt-4o-mini");
        table["api_key"] = toml_edit::value("sk-existing");
        table["url"] = toml_edit::value("https://api.openai.com/v1");

        let req = ProviderUpsertRequest {
            index: Some(0),
            provider: "openai".into(),
            model: "gpt-4.1-mini".into(),
            api_key: None,
            url: Some("https://api.openai.com/v1".into()),
            azure_deployment_id: Some(String::new()),
            azure_api_version: Some(String::new()),
        };

        apply_provider_table(&mut table, &req, &LlmProvider::Openai, true).unwrap();

        assert_eq!(table["model"].as_str(), Some("gpt-4.1-mini"));
        assert_eq!(table["api_key"].as_str(), Some("sk-existing"));
    }

    #[test]
    fn providers_response_uses_persisted_fallbacks_even_without_runtime_match() {
        let mut cfg = AppConfig::test_default();
        cfg.llm_provider = LlmProvider::Azure;
        cfg.external_llm_model = "gpt-4o-mini".into();
        cfg.external_llm_url = "https://azure.example".into();
        cfg.azure_deployment_id = "deploy-a".into();
        cfg.fallback_providers = vec![crate::config::FallbackProviderConfig {
            provider: LlmProvider::Openai,
            model: "gpt-4.1-mini".into(),
            api_key: "sk-test".into(),
            provider_keys: Vec::new(),
            key_rotation_strategy: KeyRotationStrategy::RoundRobin,
            key_cooldown_secs: 60,
            url: "https://api.openai.com/v1".into(),
            azure_deployment_id: String::new(),
            azure_api_version: DEFAULT_AZURE_API_VERSION.into(),
        }];

        let runtime = ProviderStatusResponse {
            active_provider: "azure".into(),
            providers: vec![ProviderStatusEntry {
                name: "azure".into(),
                active: true,
                status: ProviderHealthStatus::Healthy,
                model: "deploy-a".into(),
                raw_model: Some("gpt-4o-mini".into()),
                endpoint: "https://azure.example/openai/deployments/deploy-a/chat/completions?api-version=2024-08-01-preview".into(),
                config_url: Some("https://azure.example".into()),
                api_key_configured: true,
                endpoint_configured: true,
                config_index: None,
                azure_deployment_id: Some("deploy-a".into()),
                azure_api_version: Some(DEFAULT_AZURE_API_VERSION.into()),
                requests_total: 1,
                errors_total: 0,
                key_rotation_strategy: "round_robin".into(),
                key_cooldown_secs: 60,
                keys: Vec::new(),
                last_success: None,
                last_error: None,
                last_error_message: None,
            }],
        };

        let response =
            build_dashboard_provider_response(&cfg, runtime, &std::collections::HashMap::new());

        assert_eq!(response.providers.len(), 2);
        let fallback = response
            .providers
            .iter()
            .find(|entry| !entry.active)
            .expect("fallback provider should be present");
        assert_eq!(fallback.name, "openai");
        assert_eq!(fallback.config_index, Some(0));
        assert_eq!(fallback.raw_model.as_deref(), Some("gpt-4.1-mini"));
    }

    #[test]
    fn providers_response_uses_probe_health_for_persisted_provider_without_runtime_match() {
        let mut cfg = AppConfig::test_default();
        cfg.llm_provider = LlmProvider::Openai;
        cfg.external_llm_model = "gpt-4o-mini".into();
        cfg.external_llm_url = "https://api.openai.com/v1".into();

        let mut health_states = std::collections::HashMap::new();
        health_states.insert(
            "openai::gpt-4o-mini".into(),
            ProviderHealthStateSnapshot {
                requests_total: 0,
                errors_total: 0,
                last_success: Some("2026-04-08T12:00:00Z".into()),
                last_error: None,
                last_error_message: None,
                status: ProviderHealthStatus::Healthy,
            },
        );

        let response = build_dashboard_provider_response(
            &cfg,
            ProviderStatusResponse {
                active_provider: "openai".into(),
                providers: Vec::new(),
            },
            &health_states,
        );

        assert_eq!(response.providers.len(), 1);
        let provider = &response.providers[0];
        assert_eq!(provider.status, ProviderHealthStatus::Healthy);
        assert_eq!(
            provider.last_success.as_deref(),
            Some("2026-04-08T12:00:00Z")
        );
        assert_eq!(provider.requests_total, 0);
        assert_eq!(provider.errors_total, 0);
    }
}

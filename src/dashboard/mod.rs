//! Embedded web management dashboard.
//!
//! Serves a single-page application at `/dashboard` and exposes a set of
//! authenticated JSON endpoints under `/api/admin/` that the frontend
//! consumes.  All admin API routes reuse the same `AppState` and API-key
//! auth middleware as the rest of the gateway.

use std::sync::Arc;

use axum::extract::Request;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get};
use serde::{Deserialize, Serialize};
use toml_edit::DocumentMut;

use crate::models::UsageStatsResponse;
use crate::state::AppState;

// Embedded frontend — compiled into the binary at build time.
const DASHBOARD_HTML: &str = include_str!("index.html");
const LOGO_PNG: &[u8] = include_bytes!("logo.png");

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
    pub total_requests: u64,
    pub total_deflected: u64,
    pub deflection_rate: f64,
    pub total_tokens: u64,
    pub estimated_cost_usd: f64,
    pub estimated_saved_cost_usd: f64,
    pub request_logging_enabled: bool,
    pub slm_router_enabled: bool,
    pub offline_mode: bool,
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

    let overview = OverviewResponse {
        version: env!("CARGO_PKG_VERSION"),
        provider: primary.provider_name().to_string(),
        model: primary.configured_model_id().to_string(),
        total_requests: total_req,
        total_deflected: usage.total_deflected_requests,
        deflection_rate: usage.deflection_rate,
        total_tokens: usage.total_tokens,
        estimated_cost_usd: usage.estimated_cost_usd,
        estimated_saved_cost_usd: usage.estimated_saved_cost_usd,
        request_logging_enabled: state.config.enable_request_logs,
        slm_router_enabled: state.config.enable_slm_router,
        offline_mode: state.config.offline_mode,
    };

    (StatusCode::OK, Json(overview)).into_response()
}

// ── Providers endpoint ────────────────────────────────────────────────

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

    let resp = state.provider_status();
    (StatusCode::OK, Json(resp)).into_response()
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

// ── Recent requests endpoint ──────────────────────────────────────────

const RECENT_REQUESTS_LIMIT: usize = 50;

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

    let cfg = &state.config;
    let view = ConfigView {
        config_file: CONFIG_FILE.to_string(),
        file_exists: std::path::Path::new(CONFIG_FILE).exists(),
        host_port: cfg.host_port.clone(),
        proxy_port: cfg.proxy_port.clone(),
        offline_mode: cfg.offline_mode,
        llm_provider: cfg.llm_provider.as_str().to_string(),
        external_llm_model: cfg.external_llm_model.clone(),
        external_llm_url: cfg.external_llm_url.clone(),
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
    };
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
            Json(
                serde_json::json!({"error": "similarity_threshold must be between 0.0 and 1.0"}),
            ),
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

    // Notify about restart requirement (we can't hot-reload AppConfig).
    let _ = &state.config; // just to satisfy the borrow; live reload is out of scope.

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
        .route("/api/admin/usage", get(admin_usage_handler))
        .route("/api/admin/requests", get(admin_requests_handler))
        .route(
            "/api/admin/config",
            get(admin_config_get_handler).post(admin_config_post_handler),
        )
}

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
use serde::Serialize;

use crate::models::UsageStatsResponse;
use crate::state::AppState;

// Embedded frontend — compiled into the binary at build time.
const DASHBOARD_HTML: &str = include_str!("index.html");

// ── Static asset ──────────────────────────────────────────────────────

/// Serve the dashboard SPA (no auth — the shell is static HTML).
pub async fn dashboard_index() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        DASHBOARD_HTML,
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

// ── Router builders ───────────────────────────────────────────────────

/// Unauthenticated router — serves the dashboard shell HTML.
pub fn dashboard_static_router() -> Router {
    Router::new()
        .route(
            "/dashboard",
            get(|| async { axum::response::Redirect::permanent("/dashboard/") }),
        )
        .route("/dashboard/", get(dashboard_index))
}

/// Authenticated router — all endpoints require the gateway API key.
/// Caller is responsible for layering auth middleware.
pub fn admin_api_routes() -> Router {
    Router::new()
        .route("/api/admin/overview", get(admin_overview_handler))
        .route("/api/admin/providers", get(admin_providers_handler))
        .route("/api/admin/usage", get(admin_usage_handler))
        .route("/api/admin/requests", get(admin_requests_handler))
}

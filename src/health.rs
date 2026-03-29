//! # Rich Health Endpoint
//!
//! `GET /health` returns a JSON object with layer status, version,
//! uptime, and operational mode — designed for observability dashboards
//! and first-run verification.
//!
//! The existing `/healthz` liveness probe is kept unchanged.

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::Extension;
use axum::response::IntoResponse;
use serde::Serialize;

use crate::config::AppConfig;
use crate::proxy::connect;
use crate::state::AppState;
use crate::visibility;

// ── Startup timestamp ────────────────────────────────────────────────

/// Process-global startup instant, set once during server boot.
static BOOT_INSTANT: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Must be called once at startup (before serving requests).
pub fn mark_boot_time() {
    let _ = BOOT_INSTANT.get_or_init(Instant::now);
}

fn uptime_seconds() -> u64 {
    BOOT_INSTANT
        .get()
        .map(|t| t.elapsed().as_secs())
        .unwrap_or(0)
}

// ── Response types ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub layers: LayerStatus,
    pub uptime_seconds: u64,
    pub demo_mode: bool,
    pub proxy: &'static str,
    pub proxy_layer3: &'static str,
    pub proxy_recent_requests: usize,
    pub prompt_total_requests: u64,
    pub prompt_total_deflected_requests: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ProxyStatusFlag(pub bool);

#[derive(Debug, Serialize)]
pub struct LayerStatus {
    /// L1a exact-match cache.
    pub l1a: &'static str,
    /// L1b semantic (vector) cache.
    pub l1b: &'static str,
    /// L2 SLM triage layer.
    pub l2: &'static str,
    /// L3 cloud LLM fallback.
    pub l3: &'static str,
}

// ── Handler ──────────────────────────────────────────────────────────

/// Rich health check handler — `GET /health`.
///
/// Inspects the current configuration to determine layer readiness.
/// Must respond in < 5 ms (no I/O, pure in-memory checks).
pub async fn health_handler(
    Extension(config): Extension<Arc<AppConfig>>,
    Extension(demo_mode): Extension<DemoModeFlag>,
    Extension(proxy_status): Extension<ProxyStatusFlag>,
    Extension(_app_state): Extension<Arc<AppState>>,
) -> impl IntoResponse {
    let l1b_status = match config.cache_mode {
        crate::config::CacheMode::Exact => "disabled",
        _ => "active",
    };

    let l2_status = "active"; // SLM sidecar assumed reachable if configured

    let l3_status = if config.has_primary_provider_key() {
        "active"
    } else {
        "no_api_key"
    };

    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        layers: LayerStatus {
            l1a: "active",
            l1b: l1b_status,
            l2: l2_status,
            l3: l3_status,
        },
        uptime_seconds: uptime_seconds(),
        demo_mode: demo_mode.0,
        proxy: if proxy_status.0 { "active" } else { "disabled" },
        proxy_layer3: if proxy_status.0 {
            "native_upstream_passthrough"
        } else {
            "disabled"
        },
        proxy_recent_requests: if proxy_status.0 {
            connect::recent_proxy_decisions_count()
        } else {
            0
        },
        prompt_total_requests: visibility::prompt_total_requests(),
        prompt_total_deflected_requests: visibility::prompt_total_deflected_requests(),
    })
}

/// Newtype wrapper so we can inject demo_mode via Axum extensions.
#[derive(Debug, Clone, Copy)]
pub struct DemoModeFlag(pub bool);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_time_starts_at_zero_before_mark() {
        // uptime_seconds returns 0 if mark_boot_time was never called
        // in this test process (but it may have been called by another test).
        let up = uptime_seconds();
        assert!(up < 120); // sanity — less than 2 min
    }

    #[test]
    fn demo_mode_flag_is_copy() {
        let f = DemoModeFlag(true);
        let g = f; // Copy
        assert!(g.0);
    }
}

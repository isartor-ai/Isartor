// =============================================================================
// tests/common/gateway.rs — Full firewall router builder for integration tests.
//
// Assembles the entire middleware stack in the same order as main.rs:
//   state injection → body buffer → monitoring → auth → cache → SLM triage → handler
// =============================================================================

use std::sync::Arc;

use axum::{extract::Request, middleware as axum_mw, routing::post, Router};

use isartor::handler::chat_handler;
use isartor::middleware::auth::auth_middleware;
use isartor::middleware::body_buffer::buffer_body_middleware;
use isartor::middleware::cache::cache_middleware;
use isartor::middleware::monitoring::root_monitoring_middleware;
use isartor::middleware::slm_triage::slm_triage_middleware;
use isartor::state::AppState;

/// Build the full firewall router with the complete middleware stack.
///
/// Layer order (outermost → innermost):
/// 1. State injection (adds `Arc<AppState>` to extensions)
/// 2. Body buffer (snapshots the body for re-reading)
/// 3. Root monitoring (start timer / OTel span)
/// 4. Auth (API key check)
/// 5. Cache middleware (L1a exact, L1b semantic)
/// 6. SLM triage (L2 classification + short-circuit)
/// 7. Handler (L3 cloud LLM fallback)
pub fn full_gateway(state: Arc<AppState>) -> Router {
    let state_for_ext = state.clone();
    Router::new()
        .route("/api/chat", post(chat_handler))
        .layer(axum_mw::from_fn(slm_triage_middleware))
        .layer(axum_mw::from_fn(cache_middleware))
        .layer(axum_mw::from_fn(auth_middleware))
        .layer(axum_mw::from_fn(root_monitoring_middleware))
        .layer(axum_mw::from_fn(buffer_body_middleware))
        .layer(axum_mw::from_fn(
            move |mut req: Request, next: axum_mw::Next| {
                let st = state_for_ext.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            },
        ))
}

/// Build the firewall with only cache + handler layers (no auth, no SLM triage).
/// Useful for isolated cache-flow tests.
pub fn cache_only_gateway(state: Arc<AppState>) -> Router {
    let state_for_ext = state.clone();
    Router::new()
        .route("/api/chat", post(chat_handler))
        .layer(axum_mw::from_fn(cache_middleware))
        .layer(axum_mw::from_fn(buffer_body_middleware))
        .layer(axum_mw::from_fn(
            move |mut req: Request, next: axum_mw::Next| {
                let st = state_for_ext.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            },
        ))
}

/// Build the firewall with SLM triage + handler (no auth, no cache).
/// Useful for isolated SLM-flow tests.
pub fn slm_only_gateway(state: Arc<AppState>) -> Router {
    let state_for_ext = state.clone();
    Router::new()
        .route("/api/chat", post(chat_handler))
        .layer(axum_mw::from_fn(slm_triage_middleware))
        .layer(axum_mw::from_fn(buffer_body_middleware))
        .layer(axum_mw::from_fn(
            move |mut req: Request, next: axum_mw::Next| {
                let st = state_for_ext.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            },
        ))
}

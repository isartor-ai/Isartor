// =============================================================================
// tests/integration/pretooluse_hook_tests.rs
//
// Integration tests for the POST /api/v1/hook/pretooluse endpoint.
// This is a public route (no auth middleware) that logs Copilot CLI tool calls
// and always returns { "action": "allow", ... }.
// =============================================================================

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::routing::post;
use http_body_util::BodyExt;
use tower::ServiceExt;

use isartor::handler::pretooluse_hook_handler;

/// Build a minimal router with only the pretooluse hook route (no middleware).
fn pretooluse_router() -> Router {
    Router::new().route("/api/v1/hook/pretooluse", post(pretooluse_hook_handler))
}

#[tokio::test]
async fn pretooluse_hook_returns_allow_for_valid_request() {
    let app = pretooluse_router();

    let payload = serde_json::json!({
        "tool": "run_command",
        "args": "echo hello",
        "timestamp": "2024-01-01T00:00:00Z"
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/hook/pretooluse")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["action"], "allow");
    assert_eq!(json["cached"], false);
    assert_eq!(json["logged"], true);
    assert!(json["reason"].is_null());
    assert!(json["result"].is_null());
}

#[tokio::test]
async fn pretooluse_hook_returns_allow_for_empty_body() {
    let app = pretooluse_router();

    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/hook/pretooluse")
        .header("content-type", "application/json")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["action"], "allow");
    assert_eq!(json["logged"], true);
}

#[tokio::test]
async fn pretooluse_hook_returns_allow_for_malformed_json() {
    let app = pretooluse_router();

    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/hook/pretooluse")
        .header("content-type", "application/json")
        .body(Body::from("not valid json {{{"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["action"], "allow");
    assert_eq!(json["logged"], true);
}

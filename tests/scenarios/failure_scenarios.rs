// =============================================================================
// tests/scenarios/failure_scenarios.rs
//
// Additional failure scenarios testing resilience of the gateway under
// various partial-failure conditions.
// =============================================================================

use std::sync::Arc;

use axum::body::Body;
use axum::extract::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common;
use crate::common::*;
use isartor::config::CacheMode;

// ═══════════════════════════════════════════════════════════════════════
// Network / Timeout Failures
// ═══════════════════════════════════════════════════════════════════════

/// SLM sidecar returns HTTP 500 — should not crash, falls through.
#[tokio::test]
async fn slm_returns_500() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config_slm_enabled(CacheMode::Exact, &mock_server.uri());
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::full_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("test"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3, "Should fall through to Layer 3");
}

/// SLM returns valid JSON but not a chat-completion structure.
#[tokio::test]
async fn slm_returns_unexpected_json() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "unexpected": true })),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config_slm_enabled(CacheMode::Exact, &mock_server.uri());
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::full_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("test"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3, "Unexpected JSON should fall through");
}

// ═══════════════════════════════════════════════════════════════════════
// Edge-Case Inputs
// ═══════════════════════════════════════════════════════════════════════

/// Empty body — the gateway should handle it without crashing.
#[tokio::test]
async fn empty_body_does_not_crash() {
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::full_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("X-API-Key", "test-key")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // The handler interprets an empty body as an empty prompt — it should
    // not panic or return 500.
    assert!(
        resp.status() == 200 || resp.status() == 400,
        "Empty body should be handled gracefully, got {}",
        resp.status()
    );
}

/// Very long prompt — no truncation errors.
#[tokio::test]
async fn very_long_prompt_handled() {
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::full_gateway(state);

    let long_prompt = "a".repeat(100_000);
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body(&long_prompt))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3);
}

/// Unicode / emoji prompt — no encoding errors.
#[tokio::test]
async fn unicode_prompt_handled() {
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::full_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("Ünïcödé 🚀 テスト"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["message"], "echo: Ünïcödé 🚀 テスト");
}

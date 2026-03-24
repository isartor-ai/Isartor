// =============================================================================
// tests/integration/end_to_end_tests.rs
//
// Full end-to-end integration tests exercising the complete Deflection Stack:
//   auth → cache → SLM triage → handler
// =============================================================================

use std::sync::Arc;

use axum::extract::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common;
use crate::common::*;
use isartor::config::CacheMode;

/// End-to-end: TEMPLATE prompt → SLM short-circuit → response cached
/// → second request hits cache at Layer 1.
#[tokio::test]
async fn e2e_simple_prompt_cached_after_slm() {
    let mock_server = MockServer::start().await;

    // Classification: TEMPLATE (tiered mode)
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json("TEMPLATE")))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    // SLM answer
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_completion_json("Hello world!")),
        )
        .mount(&mock_server)
        .await;

    let config = test_config_slm_enabled(CacheMode::Exact, &mock_server.uri());
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);

    // 1st request: SLM handles it (Layer 2).
    let app = common::gateway::full_gateway(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("hi"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 2);

    // 2nd request with same prompt → should hit exact cache at Layer 1.
    let app2 = common::gateway::full_gateway(state.clone());
    let req2 = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("hi"))
        .unwrap();

    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), 200);

    let body2 = resp2.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body2);
    // The cached response should contain the SLM's answer.
    assert!(
        text.contains("Hello world!"),
        "Second request should return the cached Layer 2 response"
    );
}

/// End-to-end: COMPLEX prompt → falls through cache + SLM → reaches L3 handler.
#[tokio::test]
async fn e2e_complex_prompt_reaches_layer_3() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json("COMPLEX")))
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
        .body(json_body("write a Rust web server"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3);
    assert_eq!(json["message"], "echo: write a Rust web server");
}

/// End-to-end: LLM failure (Layer 3) returns 502.
#[tokio::test]
async fn e2e_llm_failure_returns_502() {
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(FailAgent("provider outage")), config, embedder);
    let app = common::gateway::full_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("fail me"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 502);
}

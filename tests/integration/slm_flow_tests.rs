// =============================================================================
// tests/integration/slm_flow_tests.rs
//
// Integration tests for SLM triage middleware (Layer 2):
//   - SIMPLE classification → short-circuit with SLM answer
//   - COMPLEX classification → pass-through to Layer 3
//   - SLM unreachable → graceful fallthrough to Layer 3
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

// ═══════════════════════════════════════════════════════════════════════
// SLM Classification Tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn simple_classification_short_circuits_at_layer_2() {
    let mock_server = MockServer::start().await;

    // First call: classification → TEMPLATE (tiered mode default)
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json("TEMPLATE")))
        .up_to_n_times(1)
        .expect(1)
        .mount(&mock_server)
        .await;

    // Second call: SLM generates the answer.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_completion_json("42 is the answer")),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config_slm_enabled(CacheMode::Exact, &mock_server.uri());
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::slm_only_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("what is 6 * 7?"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["layer"], 2,
        "Simple tasks should be handled at Layer 2"
    );
    assert_eq!(json["message"], "42 is the answer");
}

#[tokio::test]
async fn complex_classification_reaches_layer_3() {
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
    let app = common::gateway::slm_only_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("implement a Rust compiler"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3, "Complex tasks should reach Layer 3");
}

#[tokio::test]
async fn slm_unreachable_falls_through_to_layer_3() {
    // Sidecar URL points to a closed port.
    let config = test_config_slm_enabled(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::slm_only_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("hello"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["layer"], 3,
        "Unreachable SLM should fall through to Layer 3"
    );
}

#[tokio::test]
async fn malformed_slm_response_falls_through() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not valid json"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config_slm_enabled(CacheMode::Exact, &mock_server.uri());
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::slm_only_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("test"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["layer"], 3,
        "Malformed SLM response should fall through to Layer 3"
    );
}

#[tokio::test]
async fn slm_500_error_falls_through() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config_slm_enabled(CacheMode::Exact, &mock_server.uri());
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);
    let app = common::gateway::slm_only_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("test"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3, "SLM 500 should fall through to Layer 3");
}

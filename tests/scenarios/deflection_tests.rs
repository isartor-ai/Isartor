// =============================================================================
// tests/scenarios/deflection_tests.rs
//
// Scenario tests verifying that the gateway deflects ≥60% of requests
// before they reach Layer 3 (cloud LLM). This is the core cost-saving
// property of the Isartor gateway.
//
// Strategy:
//   1. Send a batch of "simple" prompts through the full middleware stack
//      with a functioning SLM sidecar.
//   2. Count how many are resolved at Layer 1 (cache) or Layer 2 (SLM).
//   3. Assert deflection rate ≥ 60%.
// =============================================================================

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use axum::extract::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common;
use crate::common::*;
use isartor::config::CacheMode;

/// Helper: create a batch of "simple" prompts that the SLM should classify as SIMPLE.
fn simple_prompts() -> Vec<&'static str> {
    vec![
        "hello",
        "what time is it",
        "how are you",
        "what is 2+2",
        "hello",       // duplicate → cache hit
        "what is 2+2", // duplicate → cache hit
        "how are you", // duplicate → cache hit
        "hello",       // duplicate → cache hit
        "good morning",
        "what is 2+2", // duplicate → cache hit
    ]
}

#[tokio::test]
async fn deflection_rate_at_least_60_percent() {
    let mock_server = MockServer::start().await;

    // SLM always classifies as SIMPLE and returns a short answer.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json("SIMPLE")))
        // First call for each unique prompt is classification.
        .mount(&mock_server)
        .await;

    let l3_calls = Arc::new(AtomicU32::new(0));
    let l3_calls_clone = l3_calls.clone();

    let counting_agent = CountingAgent {
        response: "cloud answer".into(),
        counter: l3_calls_clone,
    };

    let config = test_config_slm_enabled(CacheMode::Exact, &mock_server.uri());
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(counting_agent), config, embedder);

    let prompts = simple_prompts();
    let total = prompts.len();
    let mut resolved_before_l3 = 0u32;

    for prompt in &prompts {
        let app = common::gateway::full_gateway(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .header("X-API-Key", "test-key")
            .body(json_body(prompt))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let layer = json["layer"].as_u64().unwrap_or(99);

        if layer < 3 {
            resolved_before_l3 += 1;
        }
    }

    let deflection_rate = (resolved_before_l3 as f64 / total as f64) * 100.0;

    eprintln!(
        "Deflection rate: {resolved_before_l3}/{total} = {deflection_rate:.1}% \
         (L3 calls: {})",
        l3_calls.load(Ordering::SeqCst)
    );

    assert!(
        deflection_rate >= 60.0,
        "Deflection rate should be ≥60%, was {deflection_rate:.1}%"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Failure Scenarios
// ═══════════════════════════════════════════════════════════════════════

/// When both the SLM and the LLM are down, the gateway should return a
/// 502 error — never hang or panic.
#[tokio::test]
async fn failure_scenario_slm_and_llm_both_down() {
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(FailAgent("connection refused")), config, embedder);
    let app = common::gateway::full_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("help"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        502,
        "When everything is down the gateway should return 502"
    );
}

/// When the SLM is down but the LLM works, requests should still succeed
/// via Layer 3 fallback.
#[tokio::test]
async fn failure_scenario_slm_down_llm_ok() {
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
        .body(json_body("test"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3);
    assert_eq!(json["message"], "echo: test");
}

/// When the LLM fails but a stale cache entry exists, the gateway should
/// serve the stale response (200, not 502).
#[tokio::test]
async fn failure_scenario_stale_cache_fallback() {
    use sha2::{Digest, Sha256};

    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(FailAgent("all providers down")), config, embedder);

    // Pre-populate the exact cache with a stale response.
    let key = hex::encode(Sha256::digest(b"stale-me"));
    let stale_json = serde_json::to_string(&serde_json::json!({
        "layer": 3,
        "message": "stale answer from cache",
        "model": "gpt-4o-mini"
    }))
    .unwrap();
    state.exact_cache.put(key, stale_json);

    let app = common::gateway::full_gateway(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(json_body("stale-me"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200, "Stale cache fallback should return 200");

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(
        text.contains("stale answer from cache"),
        "Should serve the stale cached response"
    );
}

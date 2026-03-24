// =============================================================================
// tests/integration/cache_flow_tests.rs
//
// Integration tests for the cache subsystem exercised through the middleware
// stack:  exact cache miss→store→hit, semantic cache, both-mode behaviour.
// =============================================================================

use axum::extract::Request;
use http_body_util::BodyExt;
use sha2::{Digest, Sha256};
use tower::ServiceExt;

use crate::common;
use crate::common::*;

use isartor::config::CacheMode;

// ═══════════════════════════════════════════════════════════════════════
// Exact Cache Flow
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn exact_cache_miss_stores_and_hit_returns_cached() {
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(SuccessAgent("first response")), config, embedder);

    // 1st request — cache miss, reaches handler.
    let app = common::gateway::cache_only_gateway(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("cache test prompt"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["layer"], 3);

    // Verify the exact cache was populated.
    let key = hex::encode(Sha256::digest(b"native|cache test prompt"));
    assert!(
        state.exact_cache.get(&key).is_some(),
        "Exact cache should have stored the response"
    );

    // 2nd request — same prompt → cache hit.
    let app2 = common::gateway::cache_only_gateway(state.clone());
    let req2 = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("cache test prompt"))
        .unwrap();

    let resp2 = app2.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), 200);

    let body2 = resp2.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body2);
    assert!(
        text.contains("first response"),
        "Second request should return the cached response"
    );
}

#[tokio::test]
async fn exact_cache_different_prompts_are_separate() {
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);

    // First prompt.
    let app = common::gateway::cache_only_gateway(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("prompt A"))
        .unwrap();
    let _ = app.oneshot(req).await.unwrap();

    // Second prompt (different key).
    let app2 = common::gateway::cache_only_gateway(state.clone());
    let req2 = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("prompt B"))
        .unwrap();
    let _ = app2.oneshot(req2).await.unwrap();

    // Both should be in cache under different keys.
    let key_a = hex::encode(Sha256::digest(b"native|prompt A"));
    let key_b = hex::encode(Sha256::digest(b"native|prompt B"));
    assert!(state.exact_cache.get(&key_a).is_some());
    assert!(state.exact_cache.get(&key_b).is_some());
}

#[tokio::test]
async fn exact_cache_same_prompt_different_sessions_do_not_collide() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let counter = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent {
        response: "session-aware".into(),
        counter: counter.clone(),
    };
    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(agent), config, embedder);

    let req1 = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("x-thread-id", "session-a")
        .body(json_body("same prompt"))
        .unwrap();
    let resp1 = common::gateway::cache_only_gateway(state.clone())
        .oneshot(req1)
        .await
        .unwrap();
    let body1 = resp1.into_body().collect().await.unwrap().to_bytes();
    let json1: serde_json::Value = serde_json::from_slice(&body1).unwrap();
    assert_eq!(json1["layer"], 3);

    let req2 = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("x-thread-id", "session-a")
        .body(json_body("same prompt"))
        .unwrap();
    let resp2 = common::gateway::cache_only_gateway(state.clone())
        .oneshot(req2)
        .await
        .unwrap();
    let body2 = resp2.into_body().collect().await.unwrap().to_bytes();
    let json2: serde_json::Value = serde_json::from_slice(&body2).unwrap();
    assert_eq!(json2["layer"], 1);

    let req3 = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("x-thread-id", "session-b")
        .body(json_body("same prompt"))
        .unwrap();
    let resp3 = common::gateway::cache_only_gateway(state.clone())
        .oneshot(req3)
        .await
        .unwrap();
    let body3 = resp3.into_body().collect().await.unwrap().to_bytes();
    let json3: serde_json::Value = serde_json::from_slice(&body3).unwrap();
    assert_eq!(json3["layer"], 3);

    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

// ═══════════════════════════════════════════════════════════════════════
// Both-Mode Cache Flow
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn both_mode_exact_hit_short_circuits() {
    let config = test_config(CacheMode::Both, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);

    // Pre-populate exact cache.
    let key = hex::encode(Sha256::digest(b"native|pre-populated"));
    let cached_body = serde_json::to_string(&serde_json::json!({
        "layer": 1,
        "message": "from exact cache",
        "model": null
    }))
    .unwrap();
    state.exact_cache.put(key, cached_body);

    let app = common::gateway::cache_only_gateway(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("pre-populated"))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(
        text.contains("from exact cache"),
        "Should return exact-cache hit, not downstream response"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Cache Latency — Ensure cache hits are fast
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn exact_cache_hit_latency_under_5ms() {
    use std::time::Instant;

    let config = test_config(CacheMode::Exact, "http://127.0.0.1:1");
    let embedder =
        Arc::new(isartor::layer1::embeddings::TextEmbedder::new().expect("TextEmbedder init"));
    let state = build_state(Arc::new(EchoAgent), config, embedder);

    // Warm the cache.
    let app = common::gateway::cache_only_gateway(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("latency-test"))
        .unwrap();
    let _ = app.oneshot(req).await.unwrap();

    // Measure cache-hit latency.
    let app2 = common::gateway::cache_only_gateway(state.clone());
    let req2 = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(json_body("latency-test"))
        .unwrap();

    let start = Instant::now();
    let resp2 = app2.oneshot(req2).await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(resp2.status(), 200);
    assert!(
        elapsed.as_millis() < 5,
        "Exact cache hit should be under 5ms, but took {}ms",
        elapsed.as_millis()
    );
}

use std::sync::Arc;

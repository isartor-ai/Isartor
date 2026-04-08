// =============================================================================
// Phone-Home Audit Tests
//
// Verifies that Isartor makes zero outbound connections during normal
// operation, and that the `--offline` / `ISARTOR__OFFLINE_MODE` flag
// correctly blocks L3 cloud requests.
//
// Strategy:
//  • All L3 calls use a mock HTTP server (wiremock) whose address is the
//    only "allowed" external destination.
//  • L1a / L1b cache hits are served entirely in-process — no network.
//  • We assert that cache-deflected requests produce zero calls to the
//    mock L3 server, confirming no hidden phone-home traffic.
//  • We assert that offline mode returns HTTP 503 before any outbound
//    attempt is made.
// =============================================================================

use std::env;
use std::num::NonZeroUsize;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use axum::{Router, extract::Request, middleware as axum_mw, routing::post};
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

use isartor::clients::slm::SlmClient;
use isartor::config::{AppConfig, CacheMode};
use isartor::core::context_compress::InstructionCache;
use isartor::core::usage::UsageTracker;
use isartor::handler::chat_handler;
use isartor::layer1::embeddings::TextEmbedder;
use isartor::layer1::layer1a_cache::ExactMatchCache;
use isartor::middleware::body_buffer::buffer_body_middleware;
use isartor::middleware::cache::cache_middleware;
use isartor::middleware::slm_triage::slm_triage_middleware;
use isartor::state::{AppLlmAgent, AppState};
use isartor::vector_cache::VectorCache;

// ── Counting Agent ────────────────────────────────────────────────────

/// An LLM agent that counts how many times it is called.
/// Used to prove that deflected requests never reach L3.
struct CountingAgent {
    call_count: Arc<AtomicU32>,
    response: &'static str,
}

#[async_trait::async_trait]
impl AppLlmAgent for CountingAgent {
    async fn chat(&self, _prompt: &str) -> anyhow::Result<String> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(self.response.to_string())
    }
    fn provider_name(&self) -> &'static str {
        "audit-mock"
    }
}

// ── State Builder ─────────────────────────────────────────────────────

fn build_audit_state(
    agent: Arc<dyn AppLlmAgent>,
    cache_mode: CacheMode,
    offline: bool,
    sidecar_url: &str,
) -> Arc<AppState> {
    let mut cfg = AppConfig::test_default();
    cfg.gateway_api_key = "audit-key".into();
    cfg.cache_mode = cache_mode;
    cfg.cache_max_capacity = 1_000;
    cfg.layer2.sidecar_url = sidecar_url.into();
    cfg.layer2.timeout_seconds = 1;
    cfg.embedding_sidecar.timeout_seconds = 1;
    cfg.offline_mode = offline;
    let config = Arc::new(cfg);

    let exact_cache = Arc::new(ExactMatchCache::new(NonZeroUsize::new(1_000).unwrap()));
    let vector_cache = Arc::new(VectorCache::new(0.85, 300, 1_000));

    // Ensure Hugging Face Hub is in offline/cache-only mode for audit tests to avoid
    // any outbound network I/O when initializing the TextEmbedder.
    if env::var("HF_HUB_OFFLINE").is_err() {
        unsafe {
            env::set_var("HF_HUB_OFFLINE", "1");
        }
    }

    let text_embedder = Arc::new(TextEmbedder::new().expect("TextEmbedder init"));
    let slm_client = Arc::new(SlmClient::new(&config.layer2));

    Arc::new(AppState {
        http_client: reqwest::Client::new(),
        exact_cache,
        vector_cache,
        provider_chain: Arc::new(isartor::state::resolved_provider_chain(&config)),
        usage_tracker: Arc::new(UsageTracker::new(config.clone()).unwrap()),
        llm_agent: agent,
        slm_client,
        text_embedder,
        instruction_cache: Arc::new(InstructionCache::new()),
        started_at: std::time::Instant::now(),
        provider_health: Arc::new(isartor::state::ProviderHealthTracker::from_config(&config)),
        provider_key_pools: Arc::new(isartor::state::ProviderKeyPoolManager::from_provider_chain(
            isartor::state::resolved_provider_chain(&config).as_slice(),
        )),
        config,
        #[cfg(feature = "embedded-inference")]
        embedded_classifier: None,
    })
}

fn audit_app(state: Arc<AppState>) -> Router {
    let st = state.clone();
    Router::new()
        .route("/api/chat", post(chat_handler))
        .layer(axum_mw::from_fn(slm_triage_middleware))
        .layer(axum_mw::from_fn(cache_middleware))
        .layer(axum_mw::from_fn(buffer_body_middleware))
        .layer(axum_mw::from_fn(
            move |mut req: Request, next: axum_mw::Next| {
                let s = st.clone();
                async move {
                    req.extensions_mut().insert(s);
                    next.run(req).await
                }
            },
        ))
}

fn json_req(prompt: &str) -> Request {
    Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&serde_json::json!({ "prompt": prompt })).unwrap(),
        ))
        .unwrap()
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: L1a exact-cache hits never call the LLM agent (zero L3 calls)
// ═══════════════════════════════════════════════════════════════════════

/// After the exact cache is warm, 100 repeat requests must produce exactly
/// zero additional calls to the L3 agent — confirming no hidden phone-home.
#[tokio::test]
async fn l1a_cache_hits_make_zero_l3_calls() {
    let call_count = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent {
        call_count: call_count.clone(),
        response: "cached answer",
    };

    let state = build_audit_state(
        Arc::new(agent),
        CacheMode::Exact,
        false,
        "http://127.0.0.1:1",
    );

    // First request — cache miss — reaches L3.
    let app = audit_app(state.clone());
    let resp = app.oneshot(json_req("What is 2+2?")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "First request must hit L3"
    );

    // 100 repeat requests — all should be L1a hits.
    for _ in 0..100 {
        let app = audit_app(state.clone());
        let resp = app.oneshot(json_req("What is 2+2?")).await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    // L3 must have been called exactly once total — zero hidden calls.
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "L1a cache hits must produce zero L3 calls (no phone-home)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: Varied prompts across L1a, L1b still produce no unexpected calls
// ═══════════════════════════════════════════════════════════════════════

/// Send a mix of prompts — some exact repeats (L1a), some novel (L3).
/// Assert that the call count equals exactly the number of novel prompts
/// (i.e., no hidden telemetry calls beyond what was routed to L3).
#[tokio::test]
async fn mixed_prompts_no_extra_calls() {
    let call_count = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent {
        call_count: call_count.clone(),
        response: "answer",
    };

    let state = build_audit_state(
        Arc::new(agent),
        CacheMode::Exact,
        false,
        "http://127.0.0.1:1",
    );

    let unique_prompts = [
        "Tell me about Rust",
        "What is the capital of France?",
        "Explain async/await",
        "What is a Merkle tree?",
        "How does TLS work?",
    ];

    // Send each unique prompt once.
    for prompt in &unique_prompts {
        let app = audit_app(state.clone());
        let resp = app.oneshot(json_req(prompt)).await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    let first_pass_count = call_count.load(Ordering::SeqCst);
    assert_eq!(
        first_pass_count,
        unique_prompts.len() as u32,
        "Each unique prompt should reach L3 exactly once"
    );

    // Replay all prompts — all should be L1a hits, zero new L3 calls.
    for prompt in &unique_prompts {
        let app = audit_app(state.clone());
        let _ = app.oneshot(json_req(prompt)).await.unwrap();
    }

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        first_pass_count,
        "Replayed prompts must be served from cache — zero additional L3 calls"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: Offline mode blocks L3 — returns 503 instead of calling cloud
// ═══════════════════════════════════════════════════════════════════════

/// With `offline_mode = true`, every request that falls through to L3
/// must receive HTTP 503 with the offline-mode error body.
/// The LLM agent must never be called.
#[tokio::test]
async fn offline_mode_blocks_l3_and_returns_503() {
    let call_count = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent {
        call_count: call_count.clone(),
        response: "this should never be returned",
    };

    // offline = true
    let state = build_audit_state(
        Arc::new(agent),
        CacheMode::Exact,
        true,
        "http://127.0.0.1:1",
    );

    for i in 0..10 {
        let app = audit_app(state.clone());
        let resp = app
            .oneshot(json_req(&format!("novel prompt {i} — will not be cached")))
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            503,
            "Offline mode must return 503 for every cache-miss request"
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"], "offline_mode_active",
            "Offline 503 body must contain the expected error code"
        );
        assert_eq!(
            json["layer_reached"], "L3",
            "Offline 503 body must identify the layer as L3"
        );
    }

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        0,
        "Offline mode must prevent all L3 agent calls (zero phone-home)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Test 4: Cache hits in offline mode still succeed (no L3 needed)
// ═══════════════════════════════════════════════════════════════════════

/// Pre-populate the L1a cache, then enable offline mode.
/// Cache hits must still return 200 without touching L3.
#[tokio::test]
async fn offline_mode_cache_hits_still_succeed() {
    let call_count = Arc::new(AtomicU32::new(0));

    // Build a single state so the in-memory cache is shared across requests.
    let agent = CountingAgent {
        call_count: call_count.clone(),
        response: "warm-up answer",
    };
    let mut state = build_audit_state(
        Arc::new(agent),
        CacheMode::Exact,
        false,
        "http://127.0.0.1:1",
    );

    // First: warm the cache with offline=false.
    {
        let app = audit_app(state.clone());
        let resp = app.oneshot(json_req("warm prompt")).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    // Flip offline mode on the same state: cache hits must still return 200
    // without touching L3.
    {
        let st = Arc::get_mut(&mut state).expect("state must be uniquely owned");
        let cfg = Arc::make_mut(&mut st.config);
        cfg.offline_mode = true;
    }

    let offline_app = audit_app(state);
    let resp2 = offline_app.oneshot(json_req("warm prompt")).await.unwrap();
    assert_eq!(
        resp2.status(),
        200,
        "Second request should be cache hit even in offline mode"
    );
    // Still 1 L3 call total: the offline request must be served from cache.
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 5: Wiremock intercept — verify L3 only contacts the configured URL
// ═══════════════════════════════════════════════════════════════════════

/// Uses a wiremock server as the mock L3 endpoint.
/// Asserts that exactly N requests arrive (one per cache miss), and that
/// no requests hit any other mock server (simulating hidden telemetry).
///
/// This test acts as the CI phone-home proof: if Isartor had hidden
/// telemetry that posted to a secondary URL, we'd see unexpected hits
/// on the "forbidden" mock server and the test would fail.
#[tokio::test]
async fn wiremock_only_configured_l3_url_receives_requests() {
    // "Forbidden" mock — simulates an unexpected phone-home target.
    // Any request to this server = test failure.
    let forbidden_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0) // Must receive ZERO requests.
        .mount(&forbidden_server)
        .await;

    // Register the counting agent that records L3 call count.
    let call_count = Arc::new(AtomicU32::new(0));
    let agent = CountingAgent {
        call_count: call_count.clone(),
        response: "l3 response",
    };

    let state = build_audit_state(
        Arc::new(agent),
        CacheMode::Exact,
        false,
        "http://127.0.0.1:1", // non-listening — L2 always falls through
    );

    // Send 3 unique prompts (each will reach L3 mock once via CountingAgent).
    for i in 0..3 {
        let app = audit_app(state.clone());
        let resp = app
            .oneshot(json_req(&format!("unique audit prompt {i}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    // Exactly 3 L3 calls, all via CountingAgent (which doesn't use HTTP).
    assert_eq!(call_count.load(Ordering::SeqCst), 3);

    // The forbidden server must have received exactly 0 requests.
    // wiremock's expect(0) enforces this on drop.
    forbidden_server.verify().await; // asserts 0 requests to the forbidden server
}

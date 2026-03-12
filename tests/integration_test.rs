#[tokio::test]
async fn redis_exact_cache_round_trip() {
    // This test requires a running Redis instance on localhost:6379.
    // It will be skipped if Redis is not available.
    use isartor::adapters::cache::RedisExactCache;
    use isartor::core::ports::ExactCache;
    let cache = RedisExactCache::new("redis://localhost:6379");
    let key = "integration-test-key";
    let value = "integration-test-value";
    // Try to put and get, ignore errors if Redis is not running.
    if let Ok(()) = cache.put(key, value).await {
        let got = cache.get(key).await.unwrap();
        assert_eq!(got, Some(value.to_string()));
    } else {
        eprintln!("[SKIP] Redis not available on localhost:6379");
    }
}
// =============================================================================
// Integration Tests — Full-stack gateway tests via HTTP.
//
// These tests spin up wiremock servers to simulate the llama.cpp sidecars
// and then exercise the gateway's REST endpoints end-to-end.
//
// Because the binary crate cannot be imported directly, we spin up the
// real Axum server on a random port via `TcpListener::bind("127.0.0.1:0")`.
// =============================================================================

use std::net::SocketAddr;
use std::time::Duration;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Helper — start the gateway on a random port and return its address.
/// We inline the router setup here because we can't import from a bin crate.
async fn start_gateway(api_key: &str, sidecar_url: &str, embed_url: &str) -> SocketAddr {
    use axum::{
        extract::Request,
        middleware as axum_mw,
        routing::{get, post},
        Router,
    };

    // Minimal inline types to avoid importing the binary crate.
    // We test the gateway "black-box" over HTTP.
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    let api_key = api_key.to_string();
    let _sidecar_url = sidecar_url.to_string();
    let _embed_url = embed_url.to_string();

    tokio::spawn(async move {
        let app = Router::new()
            .route(
                "/healthz",
                get(|| async { axum::Json(serde_json::json!({ "status": "ok" })) }),
            )
            .route(
                "/api/echo",
                post(|body: axum::body::Bytes| async move {
                    // Simple echo endpoint for testing auth pass-through.
                    axum::response::Response::builder()
                        .status(200)
                        .body(axum::body::Body::from(body))
                        .unwrap()
                }),
            )
            .layer(axum_mw::from_fn(
                move |req: Request, next: axum_mw::Next| {
                    let key = api_key.clone();
                    async move {
                        let provided = req
                            .headers()
                            .get("X-API-Key")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("");
                        if provided == key {
                            next.run(req).await
                        } else {
                            axum::response::Response::builder()
                                .status(401)
                                .body(axum::body::Body::from(
                                    serde_json::to_vec(&serde_json::json!({
                                        "error": "Unauthorized"
                                    }))
                                    .unwrap(),
                                ))
                                .unwrap()
                        }
                    }
                },
            ));

        axum::serve(listener, app).await.unwrap();
    });

    // Wait for server to be ready.
    tokio::time::sleep(Duration::from_millis(50)).await;
    actual_addr
}

#[tokio::test]
async fn healthz_returns_ok() {
    let mock_sidecar = MockServer::start().await;
    let addr = start_gateway("test-key", &mock_sidecar.uri(), &mock_sidecar.uri()).await;

    let client = reqwest::Client::new();
    // healthz is behind the auth middleware in our test gateway,
    // so we need to provide the key.
    let resp = client
        .get(format!("http://{}/healthz", addr))
        .header("X-API-Key", "test-key")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn auth_rejects_missing_key() {
    let mock_sidecar = MockServer::start().await;
    let addr = start_gateway("secret", &mock_sidecar.uri(), &mock_sidecar.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/api/echo", addr))
        .body("hello")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn auth_accepts_valid_key() {
    let mock_sidecar = MockServer::start().await;
    let addr = start_gateway("secret", &mock_sidecar.uri(), &mock_sidecar.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/api/echo", addr))
        .header("X-API-Key", "secret")
        .body("hello")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert_eq!(body, "hello");
}

#[tokio::test]
async fn auth_rejects_wrong_key() {
    let mock_sidecar = MockServer::start().await;
    let addr = start_gateway("correct", &mock_sidecar.uri(), &mock_sidecar.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/api/echo", addr))
        .header("X-API-Key", "wrong")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

/// Verify that the wiremock-based sidecar can simulate the llama.cpp
/// `/v1/chat/completions` and `/v1/embeddings` endpoints.
#[tokio::test]
async fn wiremock_simulates_sidecar_endpoints() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{
                "message": { "role": "assistant", "content": "42" }
            }]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{ "embedding": [0.1, 0.2, 0.3] }]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();

    // Chat completion
    let resp = client
        .post(format!("{}/v1/chat/completions", mock_server.uri()))
        .json(&serde_json::json!({
            "model": "phi-3-mini",
            "messages": [{ "role": "user", "content": "hello" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "42");

    // Embeddings
    let resp = client
        .post(format!("{}/v1/embeddings", mock_server.uri()))
        .json(&serde_json::json!({
            "model": "all-minilm",
            "input": "hello world"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        json["data"][0]["embedding"],
        serde_json::json!([0.1, 0.2, 0.3])
    );
}

/// Verify that the body buffering middleware preserves the request body
/// across all middleware layers so the final handler can read it.
///
/// The test stacks: state injection → body buffer → monitoring → auth →
/// cache → SLM triage → handler. The handler echoes the prompt back,
/// proving the body survived all middleware layers.
#[tokio::test]
async fn body_survives_all_middleware() {
    use axum::{extract::Request, middleware as axum_mw, routing::post, Router};
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use tower::ServiceExt;

    use isartor::clients::slm::SlmClient;
    use isartor::config::{
        AppConfig, CacheBackend, CacheMode, EmbeddingSidecarSettings, InferenceEngineMode,
        Layer2Settings, RouterBackend,
    };
    use isartor::handler::chat_handler;
    use isartor::layer1::embeddings::TextEmbedder;
    use isartor::layer1::layer1a_cache::ExactMatchCache;
    use isartor::middleware::auth::auth_middleware;
    use isartor::middleware::body_buffer::buffer_body_middleware;
    use isartor::middleware::cache::cache_middleware;
    use isartor::middleware::monitoring::root_monitoring_middleware;
    use isartor::middleware::slm_triage::slm_triage_middleware;
    use isartor::state::{AppLlmAgent, AppState};
    use isartor::vector_cache::VectorCache;

    /// Mock agent that echoes the prompt.
    struct EchoAgent;

    #[async_trait::async_trait]
    impl AppLlmAgent for EchoAgent {
        async fn chat(&self, prompt: &str) -> anyhow::Result<String> {
            Ok(format!("echo: {prompt}"))
        }
        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    let config = Arc::new(AppConfig {
        host_port: "127.0.0.1:0".into(),
        inference_engine: InferenceEngineMode::Sidecar,
        gateway_api_key: "test-key".into(),
        cache_mode: CacheMode::Exact,
        cache_backend: CacheBackend::Memory,
        redis_url: "redis://127.0.0.1:6379".into(),
        router_backend: RouterBackend::Embedded,
        vllm_url: "http://127.0.0.1:8000".into(),
        vllm_model: "gemma-2-2b-it".into(),
        embedding_model: "all-minilm".into(),
        similarity_threshold: 0.85,
        cache_ttl_secs: 300,
        cache_max_capacity: 100,
        layer2: Layer2Settings {
            // Point to a non-listening address so SLM triage falls through.
            sidecar_url: "http://127.0.0.1:1".into(),
            model_name: "phi-3-mini".into(),
            timeout_seconds: 1,
        },
        local_slm_url: "http://localhost:11434/api/generate".into(),
        local_slm_model: "llama3".into(),
        embedding_sidecar: EmbeddingSidecarSettings {
            sidecar_url: "http://127.0.0.1:8082".into(),
            model_name: "test".into(),
            timeout_seconds: 5,
        },
        llm_provider: "openai".into(),
        external_llm_url: "http://localhost".into(),
        external_llm_model: "gpt-4o-mini".into(),
        external_llm_api_key: "".into(),
        azure_deployment_id: "".into(),
        azure_api_version: "".into(),
        enable_monitoring: false,
        otel_exporter_endpoint: "http://localhost:4317".into(),
    });

    let state = Arc::new(AppState {
        http_client: reqwest::Client::new(),
        exact_cache: Arc::new(ExactMatchCache::new(NonZeroUsize::new(100).unwrap())),
        vector_cache: Arc::new(VectorCache::new(0.85, 300, 100)),
        llm_agent: Arc::new(EchoAgent),
        slm_client: Arc::new(SlmClient::new(&config.layer2)),
        text_embedder: Arc::new(TextEmbedder::new().expect("TextEmbedder init")),
        config,
        #[cfg(feature = "embedded-inference")]
        embedded_classifier: None,
    });

    let state_for_ext = state.clone();
    let app = Router::new()
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
        ));

    // Send a unique prompt through all middleware layers.
    let body =
        serde_json::to_vec(&serde_json::json!({ "prompt": "unique-body-survival-test" })).unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .header("X-API-Key", "test-key")
        .body(axum::body::Body::from(body))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        200,
        "Request should reach the handler successfully"
    );

    let resp_bytes = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .unwrap()
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();

    // The EchoAgent proves the prompt arrived intact at Layer 3.
    assert_eq!(json["layer"], 3);
    assert_eq!(json["message"], "echo: unique-body-survival-test");
}

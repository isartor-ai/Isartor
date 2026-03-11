mod adapters;
mod clients;
mod config;
mod core;
mod factory;
mod handler;
mod layer1;
mod middleware;
mod models;
mod pipeline;
#[cfg(feature = "embedded-inference")]
mod services;
mod state;
mod telemetry;
mod vector_cache;

use std::sync::Arc;

use axum::{
    extract::Request, http::StatusCode, middleware as axum_mw, response::IntoResponse,
    routing::post, Json, Router,
};
use bytes::Bytes;
use http_body_util::BodyExt;

use crate::config::AppConfig;
use crate::pipeline::implementations::{
    embedder::LlamaCppEmbedder, external_llm::RigExternalLlm,
    intent_classifier::LlamaCppIntentClassifier, local_executor::LlamaCppLocalExecutor,
    reranker::LlamaCppReranker, vector_store::InMemoryVectorStore,
};
use crate::pipeline::{AdaptiveConcurrencyLimiter, AlgorithmSuite, ConcurrencyConfig};
use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ------------------------------------------------------------------
    // 1. Initialise structured logging & OTel telemetry
    // ------------------------------------------------------------------
    let config = Arc::new(AppConfig::load()?);
    telemetry::init_telemetry(config.clone())?;

    // ------------------------------------------------------------------
    // 2. Build shared state.
    // ------------------------------------------------------------------
        println!("Isartor gateway starting on {}", config.host_port);
        println!(
            "Cache layer configured: cache_mode={:?} embedding_model={} similarity_threshold={}",
            config.cache_mode,
            config.embedding_model,
            config.similarity_threshold
        );
        println!(
            "LLM provider configured: llm_provider={} model={}",
            config.llm_provider,
            config.external_llm_model
        );
        println!(
            "Engine & provider details: inference_engine={:?} azure_deployment_id={} external_llm_url={} api_key_len={}",
            config.inference_engine,
            config.azure_deployment_id,
            config.external_llm_url,
            config.external_llm_api_key.len()
        );

    // Initialize the in-process sentence embedder for Layer 1 semantic cache.
    // This blocks during startup (~1s) to load the ONNX model into RAM (~33 MB).
    let text_embedder = Arc::new(
        layer1::embeddings::TextEmbedder::new()
            .expect("Failed to initialize fastembed TextEmbedder (bge-small-en-v1.5)"),
    );

    let app_state = Arc::new(AppState::new(config.clone(), text_embedder));

    // ------------------------------------------------------------------
    // 3a. Build the Algorithmic Pipeline (v2) components.
    //
    //     The pipeline is an independent processing engine that runs
    //     alongside the existing middleware-based funnel. It can be
    //     accessed via /api/v2/chat.
    // ------------------------------------------------------------------
    let concurrency_limiter = Arc::new(AdaptiveConcurrencyLimiter::new(ConcurrencyConfig {
        min_concurrency: config.pipeline_min_concurrency,
        max_concurrency: config.pipeline_max_concurrency,
        target_latency: std::time::Duration::from_millis(config.pipeline_target_latency_ms),
        window_size: 100,
    }));
    let algorithm_suite = Arc::new(AlgorithmSuite {
        embedder: Box::new(LlamaCppEmbedder::new(
            app_state.http_client.clone(),
            &config.embedding_sidecar.sidecar_url,
            config.embedding_sidecar.model_name.clone(),
            config.pipeline_embedding_dim as usize,
        )),
        vector_store: Box::new(InMemoryVectorStore::new(
            config.cache_ttl_secs,
            config.cache_max_capacity,
        )),
        intent_classifier: Box::new(LlamaCppIntentClassifier::new(
            app_state.http_client.clone(),
            &config.layer2.sidecar_url,
            config.layer2.model_name.clone(),
        )),
        local_executor: Box::new(LlamaCppLocalExecutor::new(
            app_state.http_client.clone(),
            &config.layer2.sidecar_url,
            config.layer2.model_name.clone(),
        )),
        reranker: Box::new(LlamaCppReranker::new(
            app_state.http_client.clone(),
            &config.layer2.sidecar_url,
            config.layer2.model_name.clone(),
        )),
        external_llm: Box::new(RigExternalLlm::new(
            app_state.llm_agent.clone(),
            config.external_llm_model.clone(),
        )),
    });
    let pipeline_cfg = Arc::new(pipeline::PipelineConfig {
        similarity_threshold: config.pipeline_similarity_threshold,
        rerank_top_k: config.pipeline_rerank_top_k as usize,
    });

    // ------------------------------------------------------------------
    // 3b. Build the Axum router with the middleware "funnel".
    //
    //    Middleware layers execute in the order they are added via
    //    `.layer()`, but they wrap the inner handler, so the *last*
    //    `.layer()` call is the *outermost* (first to run).
    //
    //    We want execution order:
    //      Layer 0 (Auth) → Layer 1 (Cache) → Layer 2 (SLM) → Handler
    //
    //    Therefore we add them in reverse:
    //      .layer(Layer 0)   ← outermost, added last
    //      .layer(Layer 1)
    //      .layer(Layer 2)   ← innermost, added first
    // ------------------------------------------------------------------
    let state_for_ext = app_state.clone();
    let limiter_for_route = concurrency_limiter.clone();
    let suite_for_route = algorithm_suite.clone();
    let pcfg_for_route = pipeline_cfg.clone();
    // ------------------------------------------------------------------
    // 3b. Build the Axum router with the middleware "funnel".

    // Authenticated routes — go through the full middleware pipeline.
    let authenticated = Router::new()
        // Routes
        .route("/api/chat", post(handler::chat_handler))
        // v2 pipeline route — the Algorithmic AI Gateway endpoint.
        .route(
            "/api/v2/chat",
            post({
                move |request: Request| {
                    let limiter = limiter_for_route.clone();
                    let suite = suite_for_route.clone();
                    let pcfg = pcfg_for_route.clone();
                    async move {
                        // Extract the prompt from the JSON body.
                        let body_bytes: Bytes = match request.into_body().collect().await {
                            Ok(collected) => collected.to_bytes(),
                            Err(_) => {
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(serde_json::json!({
                                        "error": "Failed to read request body"
                                    })),
                                )
                                    .into_response();
                            }
                        };

                        let prompt: String =
                            serde_json::from_slice::<serde_json::Value>(&body_bytes[..])
                                .ok()
                                .and_then(|v| {
                                    v.get("prompt").and_then(|p| p.as_str()).map(String::from)
                                })
                                .unwrap_or_else(|| {
                                    String::from_utf8_lossy(&body_bytes[..]).to_string()
                                });

                        let response =
                            pipeline::execute_pipeline(prompt, &limiter, &suite, &pcfg).await;

                        let status = if response.resolved_by_layer == 0 {
                            StatusCode::SERVICE_UNAVAILABLE
                        } else {
                            StatusCode::OK
                        };

                        (status, Json(response)).into_response()
                    }
                }
            }),
        )
        // Layer 2 – SLM triage (innermost middleware).
        .layer(axum_mw::from_fn(
            middleware::slm_triage::slm_triage_middleware,
        ))
        // Layer 1 – Cache (exact / semantic / both).
        .layer(axum_mw::from_fn(middleware::cache::cache_middleware))
        // Layer 0 – Authentication (outermost functional middleware).
        .layer(axum_mw::from_fn(middleware::auth::auth_middleware))
        // Layer -1 - Root Monitoring (Top level tracing capability).
        .layer(axum_mw::from_fn(
            middleware::monitoring::root_monitoring_middleware,
        ))
        // Inject shared state into every request's extensions.
        .layer(axum_mw::from_fn(
            move |mut req: axum::extract::Request, next: axum_mw::Next| {
                let st = state_for_ext.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            },
        ));

    // Unauthenticated routes — bypass the middleware pipeline entirely.
    let public = Router::new().route("/healthz", axum::routing::get(healthz));

    let app = public.merge(authenticated);

    // ------------------------------------------------------------------
    // 4. Start the server.
    // ------------------------------------------------------------------
    let listener = tokio::net::TcpListener::bind(&config.host_port).await?;
        println!("Listening on {}", config.host_port);
    axum::serve(listener, app).await?;

    Ok(())
}

/// Simple liveness probe — returns 200 OK.
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

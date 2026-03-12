use std::sync::Arc;

use axum::{middleware as axum_mw, response::IntoResponse, routing::post, Json, Router};

use isartor::config::AppConfig;
use isartor::handler;
use isartor::middleware;
use isartor::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ------------------------------------------------------------------
    // 1. Initialise structured logging & OTel telemetry
    // ------------------------------------------------------------------
    let config = Arc::new(AppConfig::load()?);
    let _otel_guard = isartor::telemetry::init_telemetry(&config)?;

    // ------------------------------------------------------------------
    // 2. Build shared state.
    // ------------------------------------------------------------------
    tracing::info!(
        host_port = %config.host_port,
        cache_mode = ?config.cache_mode,
        embedding_model = %config.embedding_model,
        similarity_threshold = config.similarity_threshold,
        "Isartor gateway starting"
    );
    tracing::info!(
        llm_provider = %config.llm_provider,
        model = %config.external_llm_model,
        inference_engine = ?config.inference_engine,
        "LLM provider configured"
    );

    // Initialize the in-process sentence embedder for Layer 1 semantic cache.
    // This blocks during startup (~2s) to load the candle BertModel into RAM (~90 MB).
    let text_embedder = Arc::new(
        isartor::layer1::embeddings::TextEmbedder::new()
            .expect("Failed to initialize candle TextEmbedder (all-MiniLM-L6-v2)"),
    );

    let app_state = Arc::new(AppState::new(config.clone(), text_embedder));

    // ------------------------------------------------------------------
    // 3. Build the Axum router with the middleware "funnel".
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

    // Authenticated routes — go through the full middleware pipeline.
    //
    // Axum layers wrap in REVERSE order: the last `.layer()` call
    // produces the outermost (first-to-execute) middleware.
    //
    // Execution order:
    //   State injection → Body buffer → Monitoring → Auth →
    //   Cache → SLM triage → Handler
    let authenticated = Router::new()
        .route("/api/chat", post(handler::chat_handler))
        // Layer 2 – SLM triage (innermost, runs last before handler).
        .layer(axum_mw::from_fn(
            middleware::slm_triage::slm_triage_middleware,
        ))
        // Layer 1 – Cache (exact / semantic / both).
        .layer(axum_mw::from_fn(middleware::cache::cache_middleware))
        // Layer 0 – Authentication.
        .layer(axum_mw::from_fn(middleware::auth::auth_middleware))
        // Root monitoring – request-level tracing span.
        .layer(axum_mw::from_fn(
            middleware::monitoring::root_monitoring_middleware,
        ))
        // Body buffer – reads the body once and stores a BufferedBody
        // in extensions so downstream layers never consume the stream.
        .layer(axum_mw::from_fn(
            middleware::body_buffer::buffer_body_middleware,
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
    tracing::info!(addr = %config.host_port, "Listening");
    axum::serve(listener, app).await?;

    Ok(())
}

/// Simple liveness probe — returns 200 OK.
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

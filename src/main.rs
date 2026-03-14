use std::sync::Arc;

use axum::{middleware as axum_mw, response::IntoResponse, routing::post, Json, Router};
use clap::{Parser, Subcommand};
use anyhow::bail;

use isartor::config::AppConfig;
use isartor::handler;
use isartor::health::{self, DemoModeFlag};
use isartor::middleware;

#[derive(Parser)]
#[command(
    name = "isartor",
    version,
    about = "Prompt Firewall — cache-first prompt deflection"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a commented isartor.toml config scaffold and exit.
    Init,
    /// Replay bundled demo prompts against the local cache layers and print a deflection table.
    Demo,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // ── Handle `isartor init` / `isartor demo` ───────────────────
    match cli.command {
        Some(Commands::Init) => {
            isartor::first_run::write_config_scaffold()?;
            return Ok(());
        }
        Some(Commands::Demo) => {
            return run_standalone_demo().await;
        }
        None => {}
    }

    // ------------------------------------------------------------------
    // 1. Initialise structured logging & OTel telemetry
    // ------------------------------------------------------------------
    let config = Arc::new(AppConfig::load()?);
    let _otel_guard = isartor::telemetry::init_telemetry(&config)?;

    // ------------------------------------------------------------------
    // 2. Detect first-run mode
    // ------------------------------------------------------------------
    let first_run = isartor::first_run::is_first_run();
    let demo_mode = first_run;

    if first_run {
        isartor::first_run::print_welcome_banner();
    }

    // ------------------------------------------------------------------
    // 3. Build shared state.
    // ------------------------------------------------------------------
    tracing::info!(
        host_port = %config.host_port,
        cache_mode = ?config.cache_mode,
        embedding_model = %config.embedding_model,
        similarity_threshold = config.similarity_threshold,
        first_run = first_run,
        "Isartor firewall starting"
    );
    tracing::info!(
        llm_provider = %config.llm_provider,
        model = %config.external_llm_model,
        inference_engine = ?config.inference_engine,
        "LLM provider configured"
    );

    if config.enable_slm_router {
        tracing::info!(
            sidecar_url = %config.layer2.sidecar_url,
            model = %config.layer2.model_name,
            "Layer 2 SLM router enabled"
        );
    } else {
        tracing::info!("Layer 2 SLM router disabled — requests skip L2 triage");
    }

    // Initialize the in-process sentence embedder for Layer 1 semantic cache.
    // This blocks during startup (~2s) to load the candle BertModel into RAM (~90 MB).
    let text_embedder = Arc::new(
        isartor::layer1::embeddings::TextEmbedder::new()
            .expect("Failed to initialize candle TextEmbedder (all-MiniLM-L6-v2)"),
    );

    let app_state = Arc::new(isartor::state::AppState::new(config.clone(), text_embedder));

    // Mark boot time for the /health uptime counter.
    health::mark_boot_time();

    // ------------------------------------------------------------------
    // 4. Build the Axum router with the middleware Deflection Stack.
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

    // Authenticated routes — go through the full Deflection Stack.
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

    // Unauthenticated routes — bypass the Deflection Stack entirely.
    let demo_flag = DemoModeFlag(demo_mode);
    let health_config = config.clone();
    let public = Router::new()
        .route("/healthz", axum::routing::get(healthz))
        .route("/health", axum::routing::get(health::health_handler))
        .layer(axum::Extension(health_config))
        .layer(axum::Extension(demo_flag));

    let app = public.merge(authenticated);

    // ------------------------------------------------------------------
    // 5. Start the server.
    // ------------------------------------------------------------------
    let listener = tokio::net::TcpListener::bind(&config.host_port).await?;
    tracing::info!(addr = %config.host_port, "Listening");

    // ------------------------------------------------------------------
    // 6. First-run demo — runs concurrently with the server.
    // ------------------------------------------------------------------
    if first_run {
        let demo_state = app_state.clone();
        tokio::spawn(async move {
            // Brief pause so the welcome banner is visible.
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;

            match isartor::demo::run_demo(&demo_state).await {
                Ok(stats) => {
                    isartor::demo::print_demo_results(&stats);
                    if let Err(e) = isartor::demo::write_demo_result_file(&stats) {
                        tracing::warn!(error = %e, "Failed to write demo result file");
                    }
                    isartor::first_run::mark_first_run_complete();
                }
                Err(e) => {
                    tracing::error!(error = %e, "First-run demo failed");
                }
            }
        });
    }

    axum::serve(listener, app).await?;

    Ok(())
}

/// Simple liveness probe — returns 200 OK.
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// `isartor demo` — standalone demo runner.
///
/// Initialises only the caches and the in-process text embedder (no server,
/// no external LLM required), seeds the L1a/L1b layers with canonical
/// prompt/response pairs, replays the bundled 50-prompt corpus, and prints
/// a deflection summary table.
async fn run_standalone_demo() -> anyhow::Result<()> {
    // Minimal config — all defaults are fine for demo mode.
    let config = Arc::new(AppConfig::load()?);

    eprintln!();
    eprintln!("  ┌─────────────────────────────────────────────────────┐");
    eprintln!("  │  Isartor Demo — loading embedding model (~2 s) …    │");
    eprintln!("  └─────────────────────────────────────────────────────┘");
    eprintln!();

    let text_embedder = Arc::new(
        isartor::layer1::embeddings::TextEmbedder::new()
            .expect("Failed to initialize embedding model"),
    );

    let app_state = Arc::new(isartor::state::AppState::new(config, text_embedder));

    let stats = isartor::demo::run_demo(&app_state).await?;
    isartor::demo::print_demo_results(&stats);

    // Non-zero exit if deflection < 50 % so CI can catch regressions.
    if stats.deflection_pct < 50.0 {
        eprintln!(
            "  ⚠  Deflection rate {:.1}% is below the 50% acceptance threshold.",
            stats.deflection_pct
        );
        bail!(
            "Deflection rate {:.1}% is below the 50% acceptance threshold.",
            stats.deflection_pct
        );
    }

    Ok(())
}

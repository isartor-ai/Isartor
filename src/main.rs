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
    /// Enable offline / air-gap mode: all outbound cloud connections are
    /// blocked and L3 Cloud Logic is disabled.
    /// Equivalent to setting ISARTOR__OFFLINE_MODE=true.
    #[arg(long, env = "ISARTOR__OFFLINE_MODE")]
    offline: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a commented isartor.toml config scaffold and exit.
    Init,
    /// Replay bundled demo prompts against the local cache layers and print a deflection table.
    Demo,
    /// Audit what outbound connections Isartor would make with the current configuration.
    ConnectivityCheck,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // ── Handle `isartor init` / `isartor demo` / `isartor connectivity-check` ─
    match cli.command {
        Some(Commands::Init) => {
            isartor::first_run::write_config_scaffold()?;
            return Ok(());
        }
        Some(Commands::Demo) => {
            return run_standalone_demo().await;
        }
        Some(Commands::ConnectivityCheck) => {
            return run_connectivity_check().await;
        }
        None => {}
    }

    // ------------------------------------------------------------------
    // 1. Initialise structured logging & OTel telemetry
    // ------------------------------------------------------------------
    let mut config = AppConfig::load()?;

    // CLI --offline flag takes precedence over env / config file.
    if cli.offline {
        config.offline_mode = true;
    }

    let config = Arc::new(config);
    let _otel_guard = isartor::telemetry::init_telemetry(&config)?;

    // ------------------------------------------------------------------
    // 2. Detect first-run mode
    // ------------------------------------------------------------------
    let first_run = isartor::first_run::is_first_run();
    let demo_mode = first_run;

    if first_run {
        isartor::first_run::print_welcome_banner();
    }

    // Print offline mode startup status.
    if config.offline_mode {
        eprintln!();
        eprintln!("  ┌──────────────────────────────────────────────────────┐");
        eprintln!("  │  [Isartor] OFFLINE MODE ACTIVE                       │");
        eprintln!("  ├──────────────────────────────────────────────────────┤");
        eprintln!("  │  ✓ L1a Exact Cache:     active                       │");
        eprintln!("  │  ✓ L1b Semantic Cache:  active                       │");
        if config.enable_slm_router {
            eprintln!("  │  ✓ L2 SLM Router:       active                       │");
        } else {
            eprintln!("  │  - L2 SLM Router:       disabled (ISARTOR__ENABLE_SLM_ROUTER=false) │");
        }
        eprintln!("  │  ✗ L3 Cloud Logic:      DISABLED (offline mode)      │");
        eprintln!("  │  ↺ Telemetry export:    see telemetry config (external endpoints blocked in offline mode) │");
        eprintln!("  │  ✓ License validation:  offline HMAC check           │");
        eprintln!("  └──────────────────────────────────────────────────────┘");
        eprintln!();
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

/// `isartor connectivity-check` — print a connectivity audit and exit.
///
/// Shows every outbound connection endpoint Isartor would use with the
/// current configuration, so operators can verify zero unexpected
/// external connections before deploying to an air-gapped environment.
async fn run_connectivity_check() -> anyhow::Result<()> {
    let config = AppConfig::load()?;

    // L3 is considered configured only if an external LLM API key is present.
    let l3_configured = !config.external_llm_api_key.is_empty();

    let redis_configured = config.cache_backend == isartor::config::CacheBackend::Redis;
    let is_redis_internal = isartor::core::is_internal_endpoint(&config.redis_url);

    let air_gap_ok = !l3_configured || config.offline_mode;

    println!();
    println!("Isartor Connectivity Audit");
    println!("──────────────────────────");

    // L3 — cloud LLM endpoints
    println!("Required (L3 cloud routing):");
    match config.llm_provider.as_str() {
        "azure" => {
            let status = if l3_configured { "[CONFIGURED]" } else { "[NOT CONFIGURED]" };
            println!("  → {}  {}", config.external_llm_url, status);
        }
        "anthropic" => {
            let status = if l3_configured { "[CONFIGURED]" } else { "[NOT CONFIGURED]" };
            println!("  → api.anthropic.com:443  {}", status);
        }
        _ => {
            let status = if l3_configured { "[CONFIGURED]" } else { "[NOT CONFIGURED]" };
            println!("  → api.openai.com:443     {}", status);
        }
    }
    if config.offline_mode {
        println!("    (BLOCKED — offline mode active)");
    }

    // OTel — observability endpoint
    // OTel is considered configured when the endpoint is not the default localhost address.
    let otel_configured = !isartor::core::is_internal_endpoint(&config.otel_exporter_endpoint);
    println!();
    println!("Optional (observability / monitoring):");
    {
        let status = if otel_configured { "[CONFIGURED]" } else { "[NOT CONFIGURED]" };
        println!("  → {}  {}", config.otel_exporter_endpoint, status);
        if config.offline_mode && otel_configured {
            let is_ext = !isartor::core::is_internal_endpoint(&config.otel_exporter_endpoint);
            if is_ext {
                println!("    (BLOCKED — offline mode: external OTel endpoint suppressed)");
            }
        }
    }

    // Redis — internal cache
    println!();
    println!("Internal only (no external):");
    if redis_configured {
        let locality = if is_redis_internal {
            "[CONFIGURED - internal]"
        } else {
            "[CONFIGURED - external?]"
        };
        println!("  → {}  {}", config.redis_url, locality);
    } else {
        println!("  → (in-memory cache — no network connection)  [CONFIGURED - internal]");
    }

    // L2 SLM sidecar
    if config.enable_slm_router {
        println!("  → {}  [CONFIGURED - internal]", config.layer2.sidecar_url);
    }

    println!();
    println!(
        "Offline mode: {}",
        if air_gap_ok {
            "enabled (no external connections expected based on config)"
        } else {
            "disabled (L3 egress may be enabled — set ISARTOR__OFFLINE_MODE=true to block)"
        }
    );
    println!(
        "Air-gap compatible: {} {}",
        if air_gap_ok { "✓ YES" } else { "⚠ PARTIAL" },
        if air_gap_ok {
            "(L3 disabled or offline mode active)"
        } else {
            "(disable L3 or set ISARTOR__OFFLINE_MODE=true)"
        }
    );
    println!();

    Ok(())
}

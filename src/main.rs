use std::{
    ffi::OsString,
    fs::OpenOptions,
    process::{Command, Stdio},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, bail};
use axum::{
    Json, Router, middleware as axum_mw,
    response::IntoResponse,
    routing::{get, post},
};
use clap::{Parser, Subcommand};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde_json::json;

use isartor::config::{AppConfig, LlmProvider};
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

    /// Start Isartor in the background and return to the shell immediately.
    #[arg(long, global = true)]
    detach: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start Isartor in gateway-only mode or with a client-specific CONNECT proxy.
    Up(isartor::cli::up::UpArgs),
    /// Generate a commented isartor.toml config scaffold and exit.
    Init,
    /// Replay bundled demo prompts against the local cache layers and print a deflection table.
    Demo,
    /// Guided terminal setup for provider, Layer 2, connectors, and verification.
    #[command(visible_alias = "configure")]
    Setup(isartor::cli::setup::SetupArgs),
    /// Audit outbound provider connectivity and configuration.
    #[command(visible_alias = "ping", visible_alias = "connectivity-check")]
    Check,
    /// Configure local AI clients to route through Isartor.
    Connect(isartor::cli::connect::ConnectArgs),
    /// Set the API key for an LLM provider (writes to isartor.toml or env file).
    SetKey(isartor::cli::set_key::SetKeyArgs),
    /// Stop a running Isartor server.
    Stop(isartor::cli::stop::StopArgs),
    /// Update Isartor to the latest release.
    Update(isartor::cli::update::UpdateArgs),
    /// Show prompt totals, layer hits, and recent request routing.
    Stats(isartor::cli::stats::StatsArgs),
    /// Show or follow the detached Isartor log file.
    Logs(isartor::cli::logs::LogsArgs),
    /// Start a Model Context Protocol (MCP) stdio server for Copilot CLI integration.
    Mcp(isartor::cli::mcp::McpArgs),
}

const DETACH_ENV: &str = "ISARTOR_INTERNAL_DETACHED_CHILD";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Pin the rustls CryptoProvider before any TLS usage. Both `ring` and
    // `aws-lc-rs` features are enabled transitively; without an explicit
    // install rustls panics on the first handshake.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cli = Cli::parse();
    if cli.detach && is_startup_command(&cli.command) && std::env::var_os(DETACH_ENV).is_none() {
        return spawn_detached_startup();
    }

    let mut startup_mode = isartor::cli::up::StartupMode::GatewayOnly;
    let mut relaxed_provider_validation = false;

    // ── Handle `isartor init` / `isartor demo` / `isartor check` ─
    match cli.command {
        Some(Commands::Up(args)) => {
            startup_mode = args.startup_mode();
            relaxed_provider_validation =
                matches!(args.mode, Some(isartor::cli::up::UpMode::Copilot));
        }
        Some(Commands::Init) => {
            isartor::first_run::write_config_scaffold()?;
            return Ok(());
        }
        Some(Commands::Demo) => {
            return run_standalone_demo().await;
        }
        Some(Commands::Setup(args)) => {
            isartor::cli::setup::handle_setup(args).await?;
            return Ok(());
        }
        Some(Commands::Check) => {
            return run_connectivity_check().await;
        }
        Some(Commands::Connect(args)) => {
            isartor::cli::connect::handle_connect(args).await?;
            return Ok(());
        }
        Some(Commands::SetKey(args)) => {
            isartor::cli::set_key::handle_set_key(args).await?;
            return Ok(());
        }
        Some(Commands::Stop(args)) => {
            isartor::cli::stop::handle_stop(args)?;
            return Ok(());
        }
        Some(Commands::Update(args)) => {
            isartor::cli::update::handle_update(args).await?;
            return Ok(());
        }
        Some(Commands::Stats(args)) => {
            isartor::cli::stats::handle_stats(args).await?;
            return Ok(());
        }
        Some(Commands::Logs(args)) => {
            isartor::cli::logs::handle_logs(args)?;
            return Ok(());
        }
        Some(Commands::Mcp(args)) => {
            isartor::cli::mcp::handle_mcp(args).await?;
            return Ok(());
        }
        None => {}
    }

    // ------------------------------------------------------------------
    // 1. Initialise structured logging & OTel telemetry
    // ------------------------------------------------------------------
    let mut config = AppConfig::load_with_validation(!relaxed_provider_validation)?;

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

    isartor::first_run::print_startup_banner(first_run);

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
        eprintln!(
            "  │  ↺ Telemetry export:    see telemetry config (external endpoints blocked in offline mode) │"
        );
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
    tracing::info!("{}", isartor::cli::up::startup_log_line(startup_mode));

    if config.gateway_api_key.is_empty() {
        tracing::info!(
            "Gateway auth disabled (local-first default). Set ISARTOR__GATEWAY_API_KEY to enable."
        );
    } else {
        tracing::info!("Gateway auth enabled (Layer 0)");
    }

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
    // When cache_mode=exact, semantic matching is not used so we skip the download.
    let text_embedder = Arc::new(
        if matches!(config.cache_mode, isartor::config::CacheMode::Exact) {
            tracing::info!("cache_mode=exact — skipping semantic embedder download (L1b disabled)");
            isartor::layer1::embeddings::TextEmbedder::new_noop()
        } else {
            isartor::layer1::embeddings::TextEmbedder::new().map_err(|e| {
                anyhow::anyhow!(
                    "Failed to initialize candle TextEmbedder (all-MiniLM-L6-v2): {e:#}. Hint: set HF_HOME=/tmp/huggingface (or ISARTOR_HF_CACHE_DIR) to a writable path. In Docker: -e HF_HOME=/tmp/huggingface -v isartor-hf:/tmp/huggingface"
                )
            })?
        },
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
    //      Layer 0 (Auth) → Layer 1 (Cache) → Layer 2 (SLM) → Layer 2.5 (Context Optimizer) → Handler
    //
    //    Therefore we add them in reverse:
    //      .layer(Layer 0)     ← outermost, added last
    //      .layer(Layer 1)
    //      .layer(Layer 2)
    //      .layer(Layer 2.5)   ← innermost, added first
    // ------------------------------------------------------------------
    let state_for_ext = app_state.clone();

    // Authenticated routes — go through the full Deflection Stack.
    let authenticated = Router::new()
        .route("/api/chat", post(handler::chat_handler))
        // Compatibility routes for common client SDKs.
        .route("/api/v1/chat", post(handler::chat_handler))
        .route(
            "/v1/chat/completions",
            post(handler::openai_chat_completions_handler),
        )
        .route("/v1/messages", post(handler::anthropic_messages_handler))
        // Layer 2.5 – Context Optimizer (innermost, runs just before handler).
        .layer(axum_mw::from_fn(
            middleware::context_optimizer::context_optimizer_middleware,
        ))
        // Layer 2 – SLM triage (runs after cache, before context optimizer).
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

    let state_for_metadata = app_state.clone();
    let authenticated_metadata = Router::new()
        .route("/v1/models", get(handler::openai_models_handler))
        .layer(axum_mw::from_fn(middleware::auth::auth_middleware))
        .layer(axum_mw::from_fn(
            move |mut req: axum::extract::Request, next: axum_mw::Next| {
                let st = state_for_metadata.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            },
        ));

    let state_for_debug = app_state.clone();
    let debug = Router::new()
        .route(
            "/debug/proxy/recent",
            get(isartor::proxy::connect::recent_proxy_requests_handler),
        )
        .route(
            "/debug/stats/prompts",
            get(isartor::visibility::prompt_stats_handler),
        )
        .route(
            "/debug/stats/agents",
            get(isartor::visibility::agent_stats_handler),
        )
        .layer(axum_mw::from_fn(middleware::auth::auth_middleware))
        .layer(axum_mw::from_fn(
            middleware::monitoring::root_monitoring_middleware,
        ))
        .layer(axum_mw::from_fn(
            middleware::body_buffer::buffer_body_middleware,
        ))
        .layer(axum_mw::from_fn(
            move |mut req: axum::extract::Request, next: axum_mw::Next| {
                let st = state_for_debug.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            },
        ));

    // Unauthenticated routes — bypass the Deflection Stack entirely.
    let demo_flag = DemoModeFlag(demo_mode);
    let health_config = config.clone();
    let state_for_cache = app_state.clone();
    let public = Router::new()
        .route("/healthz", axum::routing::get(healthz))
        .route("/health", axum::routing::get(health::health_handler))
        .route(
            "/api/v1/hook/pretooluse",
            axum::routing::post(handler::pretooluse_hook_handler),
        )
        .route(
            "/api/v1/cache/lookup",
            axum::routing::post(handler::cache_lookup_handler),
        )
        .route(
            "/api/v1/cache/store",
            axum::routing::post(handler::cache_store_handler),
        )
        .route(
            "/mcp",
            axum::routing::get(handler::mcp_http_get_handler)
                .post(handler::mcp_http_post_handler)
                .delete(handler::mcp_http_delete_handler),
        )
        .route(
            "/mcp/",
            axum::routing::get(handler::mcp_http_get_handler)
                .post(handler::mcp_http_post_handler)
                .delete(handler::mcp_http_delete_handler),
        )
        .layer(axum_mw::from_fn(
            move |mut req: axum::extract::Request, next: axum_mw::Next| {
                let st = state_for_cache.clone();
                async move {
                    req.extensions_mut().insert(st);
                    next.run(req).await
                }
            },
        ))
        .layer(axum::Extension(app_state.clone()))
        .layer(axum::Extension(health_config))
        .layer(axum::Extension(health::ProxyStatusFlag(
            startup_mode.starts_proxy(),
        )))
        .layer(axum::Extension(demo_flag));

    let app = public
        .merge(debug)
        .merge(authenticated_metadata)
        .merge(authenticated);

    // ------------------------------------------------------------------
    // 5. Start the server and, when requested, the CONNECT proxy.
    // ------------------------------------------------------------------
    let listener = tokio::net::TcpListener::bind(&config.host_port).await?;
    tracing::info!(addr = %config.host_port, "API gateway listening");

    // Write PID file so `isartor stop` can find us.
    if let Err(e) = isartor::cli::stop::write_pid_file() {
        tracing::warn!(error = %e, "Failed to write PID file");
    }

    let proxy_handle = if startup_mode.starts_proxy() {
        let proxy_addr = config.proxy_port.clone();
        let proxy_state = app_state.clone();
        match isartor::proxy::tls::IsartorCa::load_or_generate() {
            Ok(ca) => {
                let ca = Arc::new(ca);
                tracing::info!(addr = %proxy_addr, "CONNECT proxy starting");
                Some(tokio::spawn(async move {
                    if let Err(e) =
                        isartor::proxy::connect::run_connect_proxy(&proxy_addr, ca, proxy_state)
                            .await
                    {
                        tracing::error!(error = %e, "CONNECT proxy exited with error");
                    }
                }))
            }
            Err(e) => {
                tracing::warn!(error = %e, "CONNECT proxy disabled: CA generation failed");
                None
            }
        }
    } else {
        tracing::info!(
            "CONNECT proxy not started. Use `isartor up copilot|claude|antigravity` when a client needs it."
        );
        None
    };

    isartor::cli::up::print_startup_card(&config, startup_mode);

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

    // Run API gateway (and proxy if started) until either exits.
    let api_server = axum::serve(listener, app);
    match proxy_handle {
        Some(proxy) => {
            tokio::select! {
                result = api_server => {
                    result?;
                }
                result = proxy => {
                    if let Err(e) = result {
                        tracing::error!(error = %e, "CONNECT proxy task panicked");
                    }
                }
            }
        }
        None => {
            api_server.await?;
        }
    }

    // Clean up PID file on shutdown.
    isartor::cli::stop::remove_pid_file();

    Ok(())
}

fn is_startup_command(command: &Option<Commands>) -> bool {
    matches!(command, None | Some(Commands::Up(_)))
}

fn spawn_detached_startup() -> anyhow::Result<()> {
    let exe = std::env::current_exe().context("cannot determine current executable path")?;
    let args = detached_child_args();
    let log_path = isartor::cli::up::startup_log_path()?;

    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let stdout_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let stderr_log = stdout_log
        .try_clone()
        .with_context(|| format!("failed to clone {}", log_path.display()))?;

    let mut command = Command::new(exe);
    command
        .args(args)
        .env(DETACH_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_log))
        .stderr(Stdio::from(stderr_log));

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }

    let child = command
        .spawn()
        .context("failed to start detached Isartor")?;

    eprintln!("  ✓ Isartor starting in background (PID {}).", child.id());
    eprintln!("    Logs: {}", log_path.display());
    eprintln!("    Stop: isartor stop");
    eprintln!("    Tip: isartor logs --follow");

    Ok(())
}

fn detached_child_args() -> Vec<OsString> {
    std::env::args_os()
        .skip(1)
        .filter(|arg| arg != "--detach")
        .collect()
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

    let text_embedder = Arc::new(isartor::layer1::embeddings::TextEmbedder::new().map_err(
        |e| {
            anyhow::anyhow!(
                "Failed to initialize embedding model (all-MiniLM-L6-v2): {e:#}. Hint: set HF_HOME to a writable path (e.g. /tmp/huggingface)."
            )
        },
    )?);

    let app_state = Arc::new(isartor::state::AppState::new(config, text_embedder));

    let stats = isartor::demo::run_demo(&app_state).await?;
    isartor::demo::print_demo_results(&stats);
    if let Err(e) = isartor::demo::write_demo_result_file(&stats) {
        eprintln!("  ⚠  Failed to write isartor_demo_result.txt: {e}");
    }

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

/// `isartor check` — print a connectivity audit and exit.
///
/// Shows every outbound connection endpoint Isartor would use with the
/// current configuration, so operators can verify zero unexpected
/// external connections before deploying to an air-gapped environment.
async fn run_connectivity_check() -> anyhow::Result<()> {
    let config = AppConfig::load()?;
    let l3_target = l3_connectivity_target(&config);

    let l3_configured = l3_target.is_configured();

    let redis_configured = config.cache_backend == isartor::config::CacheBackend::Redis;
    let is_redis_internal = isartor::core::is_internal_endpoint(&config.redis_url);

    let air_gap_ok = !l3_configured || config.offline_mode;

    println!();
    println!("Isartor Connectivity Audit");
    println!("──────────────────────────");

    // L3 — cloud LLM endpoints
    println!("Required (L3 cloud routing):");
    let status = if l3_configured {
        "[CONFIGURED]"
    } else {
        "[NOT CONFIGURED]"
    };
    println!("  Provider:   {}  {}", l3_target.provider, status);
    println!("  Model:      {}", l3_target.model);
    println!("  API key:    {}", l3_target.masked_key);
    println!("  Endpoint:   {}", l3_target.endpoint);
    println!(
        "  Ping:       {}",
        ping_l3_provider(&config, &l3_target).await
    );
    if config.offline_mode {
        println!("    (BLOCKED — offline mode active)");
    }

    // OTel — observability endpoint
    // OTel is considered configured when the endpoint is not the default localhost address.
    let otel_configured = !isartor::core::is_internal_endpoint(&config.otel_exporter_endpoint);
    println!();
    println!("Optional (observability / monitoring):");
    {
        let status = if otel_configured {
            "[CONFIGURED]"
        } else {
            "[NOT CONFIGURED]"
        };
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

struct L3ConnectivityTarget {
    provider: &'static str,
    model: String,
    masked_key: String,
    endpoint: String,
    external: bool,
    ping_kind: L3PingKind,
    requires_api_key: bool,
}

#[derive(Clone, Copy)]
enum L3PingKind {
    OpenAiModels,
    AzureChatCompletions,
    AnthropicMessages,
    GeminiModelInfo,
    CopilotSessionToken,
    OllamaTags,
    CohereModels,
    HuggingFaceModelInfo,
}

impl L3ConnectivityTarget {
    fn is_configured(&self) -> bool {
        !self.requires_api_key || self.masked_key != "(not configured)"
    }
}

fn l3_connectivity_target(config: &AppConfig) -> L3ConnectivityTarget {
    let model = match config.llm_provider {
        LlmProvider::Azure if !config.azure_deployment_id.trim().is_empty() => {
            format!(
                "{} (deployment; model {})",
                config.azure_deployment_id, config.external_llm_model
            )
        }
        _ => config.external_llm_model.clone(),
    };

    match config.llm_provider {
        LlmProvider::Azure => L3ConnectivityTarget {
            provider: "azure",
            model,
            masked_key: mask_secret(&config.external_llm_api_key),
            endpoint: format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                config.external_llm_url.trim_end_matches('/'),
                config.azure_deployment_id,
                config.azure_api_version
            ),
            external: !isartor::core::is_internal_endpoint(&config.external_llm_url),
            ping_kind: L3PingKind::AzureChatCompletions,
            requires_api_key: true,
        },
        LlmProvider::Anthropic => L3ConnectivityTarget {
            provider: "anthropic",
            model,
            masked_key: mask_secret(&config.external_llm_api_key),
            endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            external: true,
            ping_kind: L3PingKind::AnthropicMessages,
            requires_api_key: true,
        },
        LlmProvider::Copilot => L3ConnectivityTarget {
            provider: "copilot",
            model,
            masked_key: mask_secret(&config.external_llm_api_key),
            endpoint: isartor::providers::copilot::COPILOT_TOKEN_URL.to_string(),
            external: true,
            ping_kind: L3PingKind::CopilotSessionToken,
            requires_api_key: true,
        },
        LlmProvider::Gemini => L3ConnectivityTarget {
            provider: "gemini",
            model: config.external_llm_model.clone(),
            masked_key: mask_secret(&config.external_llm_api_key),
            endpoint: format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}",
                config.external_llm_model
            ),
            external: true,
            ping_kind: L3PingKind::GeminiModelInfo,
            requires_api_key: true,
        },
        LlmProvider::Ollama => L3ConnectivityTarget {
            provider: "ollama",
            model,
            masked_key: "(not required)".to_string(),
            endpoint: format!("{}/api/tags", config.external_llm_url.trim_end_matches('/')),
            external: !isartor::core::is_internal_endpoint(&config.external_llm_url),
            ping_kind: L3PingKind::OllamaTags,
            requires_api_key: false,
        },
        LlmProvider::Cohere => L3ConnectivityTarget {
            provider: "cohere",
            model,
            masked_key: mask_secret(&config.external_llm_api_key),
            endpoint: "https://api.cohere.ai/v1/models".to_string(),
            external: true,
            ping_kind: L3PingKind::CohereModels,
            requires_api_key: true,
        },
        LlmProvider::Huggingface => L3ConnectivityTarget {
            provider: "huggingface",
            model: config.external_llm_model.clone(),
            masked_key: mask_secret(&config.external_llm_api_key),
            endpoint: format!(
                "https://api-inference.huggingface.co/models/{}",
                config.external_llm_model
            ),
            external: true,
            ping_kind: L3PingKind::HuggingFaceModelInfo,
            requires_api_key: true,
        },
        LlmProvider::Openai => openai_models_target(
            "openai",
            model,
            &config.external_llm_api_key,
            "https://api.openai.com/v1/models",
        ),
        LlmProvider::Xai => openai_models_target(
            "xai",
            model,
            &config.external_llm_api_key,
            "https://api.x.ai/v1/models",
        ),
        LlmProvider::Mistral => openai_models_target(
            "mistral",
            model,
            &config.external_llm_api_key,
            "https://api.mistral.ai/v1/models",
        ),
        LlmProvider::Groq => openai_models_target(
            "groq",
            model,
            &config.external_llm_api_key,
            "https://api.groq.com/openai/v1/models",
        ),
        LlmProvider::Deepseek => openai_models_target(
            "deepseek",
            model,
            &config.external_llm_api_key,
            "https://api.deepseek.com/models",
        ),
        LlmProvider::Galadriel => openai_models_target(
            "galadriel",
            model,
            &config.external_llm_api_key,
            "https://api.galadriel.com/v1/models",
        ),
        LlmProvider::Hyperbolic => openai_models_target(
            "hyperbolic",
            model,
            &config.external_llm_api_key,
            "https://api.hyperbolic.xyz/v1/models",
        ),
        LlmProvider::Mira => openai_models_target(
            "mira",
            model,
            &config.external_llm_api_key,
            "https://api.mira.network/v1/models",
        ),
        LlmProvider::Moonshot => openai_models_target(
            "moonshot",
            model,
            &config.external_llm_api_key,
            "https://api.moonshot.cn/v1/models",
        ),
        LlmProvider::Openrouter => openai_models_target(
            "openrouter",
            model,
            &config.external_llm_api_key,
            "https://openrouter.ai/api/v1/models",
        ),
        LlmProvider::Perplexity => openai_models_target(
            "perplexity",
            model,
            &config.external_llm_api_key,
            "https://api.perplexity.ai/models",
        ),
        LlmProvider::Together => openai_models_target(
            "together",
            model,
            &config.external_llm_api_key,
            "https://api.together.xyz/v1/models",
        ),
    }
}

fn openai_models_target(
    provider: &'static str,
    model: String,
    api_key: &str,
    endpoint: &str,
) -> L3ConnectivityTarget {
    L3ConnectivityTarget {
        provider,
        model,
        masked_key: mask_secret(api_key),
        endpoint: endpoint.to_string(),
        external: true,
        ping_kind: L3PingKind::OpenAiModels,
        requires_api_key: true,
    }
}

fn mask_secret(secret: &str) -> String {
    let trimmed = secret.trim();
    if trimmed.is_empty() {
        return "(not configured)".to_string();
    }
    if trimmed.len() <= 8 {
        return "********".to_string();
    }
    format!("{}…{}", &trimmed[..4], &trimmed[trimmed.len() - 4..])
}

async fn ping_l3_provider(config: &AppConfig, target: &L3ConnectivityTarget) -> String {
    if config.offline_mode && target.external {
        return "SKIPPED — offline mode blocks external egress".to_string();
    }
    if target.requires_api_key && config.external_llm_api_key.trim().is_empty() {
        return "SKIPPED — API key not configured".to_string();
    }

    let timeout_secs = config.l3_timeout_secs.clamp(1, 15);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
    {
        Ok(client) => client,
        Err(err) => return format!("FAILED — could not build HTTP client: {err}"),
    };

    let result = match target.ping_kind {
        L3PingKind::OpenAiModels => client
            .get(&target.endpoint)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", config.external_llm_api_key),
            )
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::AzureChatCompletions => client
            .post(&target.endpoint)
            .header("api-key", &config.external_llm_api_key)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .json(&json!({
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1
            }))
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::AnthropicMessages => client
            .post(&target.endpoint)
            .header("x-api-key", &config.external_llm_api_key)
            .header("anthropic-version", "2023-06-01")
            .header(CONTENT_TYPE, "application/json")
            .json(&json!({
                "model": config.external_llm_model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}]
            }))
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::GeminiModelInfo => client
            .get(format!(
                "{}?key={}",
                target.endpoint, config.external_llm_api_key
            ))
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::CopilotSessionToken => {
            match isartor::providers::copilot::exchange_copilot_session_token(
                &client,
                &config.external_llm_api_key,
            )
            .await
            {
                Ok(_) => Ok("OK — session token exchange succeeded".to_string()),
                Err(err) => Err(err),
            }
        }
        L3PingKind::OllamaTags => client
            .get(&target.endpoint)
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::CohereModels => client
            .get(&target.endpoint)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", config.external_llm_api_key),
            )
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
        L3PingKind::HuggingFaceModelInfo => client
            .get(&target.endpoint)
            .header(
                AUTHORIZATION,
                format!("Bearer {}", config.external_llm_api_key),
            )
            .header(ACCEPT, "application/json")
            .send()
            .await
            .map_err(anyhow::Error::from)
            .and_then(summarize_ping_response),
    };

    match result {
        Ok(summary) => summary,
        Err(err) => format!("FAILED — {err}"),
    }
}

fn summarize_ping_response(response: reqwest::Response) -> anyhow::Result<String> {
    let status = response.status();
    if status.is_success() {
        Ok(format!("OK — HTTP {status}"))
    } else {
        Err(anyhow::anyhow!("HTTP {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_commands_are_detachable() {
        assert!(is_startup_command(&None));
        assert!(is_startup_command(&Some(Commands::Up(
            isartor::cli::up::UpArgs { mode: None },
        ))));
        assert!(!is_startup_command(&Some(Commands::Demo)));
    }

    #[test]
    fn mask_secret_censors_configured_keys() {
        assert_eq!(mask_secret(""), "(not configured)");
        assert_eq!(mask_secret("short"), "********");
        assert_eq!(mask_secret("gsk_testkey12345678"), "gsk_…5678");
    }

    #[test]
    fn connectivity_target_uses_groq_endpoint_and_model() {
        let mut config = AppConfig::load_with_validation(false).unwrap();
        config.llm_provider = LlmProvider::Groq;
        config.external_llm_model = "llama-3.1-8b-instant".into();
        config.external_llm_api_key = "gsk_testkey12345678".into();

        let target = l3_connectivity_target(&config);
        assert_eq!(target.provider, "groq");
        assert_eq!(target.model, "llama-3.1-8b-instant");
        assert_eq!(target.masked_key, "gsk_…5678");
        assert_eq!(target.endpoint, "https://api.groq.com/openai/v1/models");
    }

    #[test]
    fn connectivity_target_uses_azure_chat_completions_endpoint() {
        let mut config = AppConfig::load_with_validation(false).unwrap();
        config.llm_provider = LlmProvider::Azure;
        config.external_llm_url = "https://example.openai.azure.com".into();
        config.azure_deployment_id = "gpt-4o-mini".into();
        config.azure_api_version = "2024-08-01-preview".into();
        config.external_llm_model = "gpt-4o-mini".into();
        config.external_llm_api_key = "azure-secret-key".into();

        let target = l3_connectivity_target(&config);
        assert_eq!(target.provider, "azure");
        assert_eq!(target.model, "gpt-4o-mini (deployment; model gpt-4o-mini)");
        assert_eq!(
            target.endpoint,
            "https://example.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
        );
    }
}

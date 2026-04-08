//! `isartor auth` — interactive OAuth / API-key credential management.
//!
//! # Subcommands
//!
//! - `isartor auth login <provider>` — authenticate and store credentials
//! - `isartor auth status` — list authenticated providers
//! - `isartor auth logout <provider>` — remove stored credentials

use clap::{Args, Subcommand};
use reqwest::Client;

use crate::auth::device_flow;
use crate::auth::{AuthMethod, TokenStore, find_provider};

// ── Arg types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
pub struct AuthArgs {
    #[arg(value_name = "PROVIDER", conflicts_with = "command")]
    pub provider: Option<String>,

    #[command(subcommand)]
    pub command: Option<AuthCommand>,
}

#[derive(Debug, Subcommand)]
pub enum AuthCommand {
    /// Authenticate with a provider and store credentials.
    ///
    /// PROVIDER is one of: copilot, gemini, kiro, anthropic, openai
    Login {
        #[arg(value_name = "PROVIDER")]
        provider: String,
    },
    /// Show which providers have stored credentials.
    Status,
    /// Remove stored credentials for PROVIDER.
    Logout {
        #[arg(value_name = "PROVIDER")]
        provider: String,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn handle_auth(args: AuthArgs) -> anyhow::Result<()> {
    let store = TokenStore::open()?;

    if let Some(provider) = args.provider {
        return run_login(&store, &provider).await;
    }

    match args.command {
        Some(AuthCommand::Login { provider }) => run_login(&store, &provider).await,
        Some(AuthCommand::Status) => run_status(&store),
        Some(AuthCommand::Logout { provider }) => run_logout(&store, &provider),
        None => anyhow::bail!(
            "missing provider or subcommand; use `isartor auth <provider>`, `isartor auth status`, or `isartor auth logout <provider>`"
        ),
    }
}

// ── Login ─────────────────────────────────────────────────────────────────────

async fn run_login(store: &TokenStore, provider_name: &str) -> anyhow::Result<()> {
    let Some(provider) = find_provider(provider_name) else {
        eprintln!("Unknown provider: '{provider_name}'");
        eprintln!();
        eprintln!("Supported providers:");
        for p in crate::auth::all_providers() {
            eprintln!("  {} — {}", p.provider_name(), p.display_name());
        }
        anyhow::bail!("unknown provider '{provider_name}'");
    };

    // Already authenticated?
    if let Ok(Some(token)) = store.load(provider.provider_name()) {
        if !token.is_expired() {
            println!(
                "✓ Already authenticated with {} (use 'isartor auth logout {}' to reset).",
                provider.display_name(),
                provider.provider_name()
            );
            return Ok(());
        }
        println!(
            "Existing token for {} is expired — re-authenticating.",
            provider.display_name()
        );
    }

    let http = Client::builder()
        .user_agent(format!("isartor/{}", env!("CARGO_PKG_VERSION")))
        .build()?;

    let token = match provider.auth_method() {
        AuthMethod::DeviceFlow => {
            let state = provider.start_device_flow(&http).await?;
            device_flow::print_device_flow_instructions(provider.display_name(), &state);
            device_flow::poll_until_ready(provider.as_ref(), &http, &state).await?
        }
        AuthMethod::ApiKey => provider.prompt_api_key().await?,
        AuthMethod::BrowserOAuth => {
            anyhow::bail!(
                "{} uses browser OAuth which is not yet supported in the CLI",
                provider.display_name()
            );
        }
    };

    store.save(&token)?;
    println!(
        "✓ Authenticated with {} and credentials saved.",
        provider.display_name()
    );
    Ok(())
}

// ── Status ────────────────────────────────────────────────────────────────────

fn run_status(store: &TokenStore) -> anyhow::Result<()> {
    let authenticated = store.list_authenticated()?;

    if authenticated.is_empty() {
        println!("No providers are authenticated.");
        println!("Run `isartor auth login <provider>` to authenticate.");
        return Ok(());
    }

    println!(
        "{:<12}  {:<22}  {:<8}  Expires In",
        "Provider", "Display Name", "Status"
    );
    println!("{}", "─".repeat(62));

    for provider_name in &authenticated {
        let display = crate::auth::find_provider(provider_name)
            .map(|p| p.display_name().to_string())
            .unwrap_or_else(|| provider_name.clone());

        let (status, expires) = match store.load(provider_name) {
            Ok(Some(token)) => {
                let status = if token.is_expired() {
                    "expired"
                } else {
                    "active"
                };
                let expires = match token.expires_in_secs() {
                    None => "no expiry".to_string(),
                    Some(s) if s <= 0 => "expired".to_string(),
                    Some(s) if s < 3600 => format!("{}m", s / 60),
                    Some(s) => format!("{}h", s / 3600),
                };
                (status.to_string(), expires)
            }
            Ok(None) => ("missing".to_string(), "-".to_string()),
            Err(e) => (format!("error: {e}"), "-".to_string()),
        };

        println!(
            "{:<12}  {:<22}  {:<8}  {}",
            provider_name, display, status, expires
        );
    }

    Ok(())
}

// ── Logout ────────────────────────────────────────────────────────────────────

fn run_logout(store: &TokenStore, provider_name: &str) -> anyhow::Result<()> {
    let canonical = crate::auth::find_provider(provider_name)
        .map(|p| p.provider_name().to_string())
        .unwrap_or_else(|| provider_name.to_lowercase());

    store.delete(&canonical)?;
    println!("✓ Credentials for '{}' have been removed.", canonical);
    Ok(())
}

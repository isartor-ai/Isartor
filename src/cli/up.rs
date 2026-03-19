use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use crate::config::AppConfig;

#[derive(Parser, Debug, Clone, Default)]
pub struct UpArgs {
    #[command(subcommand)]
    pub mode: Option<UpMode>,
}

#[derive(Subcommand, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpMode {
    /// Start with GitHub Copilot CLI integration hints.
    Copilot,
    /// Start with Claude Code integration hints.
    Claude,
    /// Start with Antigravity integration hints.
    Antigravity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupMode {
    GatewayOnly,
    Proxy { client: ProxyClient },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyClient {
    Copilot,
    Claude,
    Antigravity,
}

impl UpArgs {
    pub fn startup_mode(&self) -> StartupMode {
        match self.mode {
            // Client subcommands now start gateway-only; the CONNECT proxy
            // is no longer required for client integrations (they use base
            // URL override or hooks instead).
            Some(UpMode::Copilot) => StartupMode::GatewayOnly,
            Some(UpMode::Claude) => StartupMode::GatewayOnly,
            Some(UpMode::Antigravity) => StartupMode::GatewayOnly,
            None => StartupMode::GatewayOnly,
        }
    }

    pub fn client_hint(&self) -> Option<&'static str> {
        match self.mode {
            Some(UpMode::Copilot) => Some("isartor connect copilot"),
            Some(UpMode::Claude) => Some("isartor connect claude"),
            Some(UpMode::Antigravity) => Some("isartor connect antigravity"),
            None => None,
        }
    }
}

pub fn startup_log_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".isartor").join("isartor.log"))
}

impl StartupMode {
    pub fn starts_proxy(self) -> bool {
        matches!(self, Self::Proxy { .. })
    }
}

impl ProxyClient {
    fn label(self) -> &'static str {
        match self {
            Self::Copilot => "GitHub Copilot CLI",
            Self::Claude => "Claude Code",
            Self::Antigravity => "Antigravity",
        }
    }

    fn connect_hint(self) -> &'static str {
        match self {
            Self::Copilot => "isartor connect copilot",
            Self::Claude => "isartor connect claude",
            Self::Antigravity => "isartor connect antigravity",
        }
    }

    fn activate_hint(self) -> Option<&'static str> {
        match self {
            Self::Copilot => Some("source ~/.isartor/env/copilot.sh"),
            Self::Claude => None,
            Self::Antigravity => Some("source ~/.isartor/env/antigravity.sh"),
        }
    }
}

pub fn print_startup_card(config: &AppConfig, mode: StartupMode) {
    let gateway_url = localhost_url(&config.host_port);
    let auth = if config.gateway_api_key.is_empty() {
        "disabled"
    } else {
        "enabled"
    };

    eprintln!();
    eprintln!("  ┌──────────────────────────────────────────────────────────────┐");
    eprintln!("  │  Isartor up                                                 │");
    eprintln!("  ├──────────────────────────────────────────────────────────────┤");
    eprintln!("  │  Gateway: {:<50}│", gateway_url);
    eprintln!("  │  Auth:    {:<50}│", auth);
    eprintln!(
        "  │  Shell:   {:<50}│",
        "use --detach to return immediately"
    );

    match mode {
        StartupMode::GatewayOnly => {
            eprintln!("  │  Next:    isartor connect copilot|claude|antigravity       │");
            eprintln!("  │                                                              │");
            eprintln!("  │  Endpoints:                                                  │");
            eprintln!("  │    POST /v1/chat/completions  (OpenAI format)                │");
            eprintln!("  │    POST /v1/messages          (Anthropic format)             │");
        }
        StartupMode::Proxy { client } => {
            let proxy_url = localhost_url(&config.proxy_port);
            eprintln!("  │  Proxy:   {:<50}│", proxy_url);
            eprintln!("  │  Client:  {:<50}│", client.label());
            eprintln!("  │  Next:    {:<50}│", client.connect_hint());
            if let Some(activate_hint) = client.activate_hint() {
                eprintln!("  │  Then:    {:<50}│", activate_hint);
            }
        }
    }

    eprintln!("  └──────────────────────────────────────────────────────────────┘");
    eprintln!();
}

pub fn startup_log_line(mode: StartupMode) -> &'static str {
    match mode {
        StartupMode::GatewayOnly => {
            "Gateway started. Use `isartor connect copilot|claude|antigravity` to configure clients."
        }
        StartupMode::Proxy {
            client: ProxyClient::Copilot,
        } => "Proxy mode enabled for GitHub Copilot CLI (`isartor up copilot`).",
        StartupMode::Proxy {
            client: ProxyClient::Claude,
        } => "Proxy mode enabled for Claude Code (`isartor up claude`).",
        StartupMode::Proxy {
            client: ProxyClient::Antigravity,
        } => "Proxy mode enabled for Antigravity (`isartor up antigravity`).",
    }
}

fn localhost_url(bind_addr: &str) -> String {
    let port = bind_addr
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);
    format!("http://localhost:{port}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn up_without_mode_starts_gateway_only() {
        let args = UpArgs { mode: None };
        assert_eq!(args.startup_mode(), StartupMode::GatewayOnly);
    }

    #[test]
    fn up_with_client_starts_gateway() {
        let args = UpArgs {
            mode: Some(UpMode::Copilot),
        };
        // Client subcommands now start gateway-only (no proxy needed).
        assert_eq!(args.startup_mode(), StartupMode::GatewayOnly);
    }

    #[test]
    fn localhost_url_maps_bind_address_to_localhost() {
        assert_eq!(localhost_url("0.0.0.0:8081"), "http://localhost:8081");
    }
}

use clap::Parser;

use super::{DEFAULT_GATEWAY_URL, state::ConnectionState, test_isartor_connection};
use crate::config::AppConfig;
use crate::models::ProxyRouteDecision;

#[derive(Parser, Debug, Clone)]
pub struct StatusArgs {
    /// Isartor gateway URL (default: http://localhost:8080)
    #[arg(long, default_value = DEFAULT_GATEWAY_URL)]
    pub gateway_url: String,

    /// Gateway API key (optional). If omitted, status will still check /health.
    #[arg(long, env = "ISARTOR__GATEWAY_API_KEY")]
    pub gateway_api_key: Option<String>,

    /// Number of recent proxied client requests to show.
    #[arg(long, default_value_t = 5)]
    pub proxy_recent_limit: usize,
}

pub async fn handle_status(args: StatusArgs) {
    let state = ConnectionState::load();

    let gateway = args.gateway_url.trim_end_matches('/').to_string();

    println!("\nIsartor Gateway");
    match check_isartor_health(&gateway).await {
        Some(h) => {
            println!("  URL:     {}", gateway);
            println!(
                "  Status:  ✓ running (v{}, uptime {}s)",
                h.version, h.uptime_seconds
            );
            println!(
                "  Layers:  L1a {}  L1b {}  L2 {}  L3 {}",
                layer_icon(h.layers.l1a == "active"),
                layer_icon(h.layers.l1b == "active"),
                layer_icon(h.layers.l2 == "active"),
                layer_icon(h.layers.l3 == "active"),
            );
            println!(
                "  Proxy:   {} (L3 via {}, recent {})",
                h.proxy, h.proxy_layer3, h.proxy_recent_requests
            );
        }
        None => {
            println!("  URL:     {}", gateway);
            println!("  Status:  ✗ not running");
            println!("  Start:   docker run -p 8080:8080 ghcr.io/isartor-ai/isartor:latest");
        }
    }

    println!("\nConnected Clients");

    let all_clients = [
        "copilot",
        "copilot-vscode",
        "claude",
        "claude-copilot",
        "openclaw",
        "opencode",
        "antigravity",
    ];
    for client in all_clients {
        match state.connections.get(client) {
            Some(conn) => {
                println!("  ✓ {}", client_display_name(client));
                println!("      Method: {}", integration_method(client));
                for file in &conn.config_files_modified {
                    println!("      Config: {}", file);
                }

                let test = test_isartor_connection(
                    &conn.gateway_url,
                    args.gateway_api_key.as_deref(),
                    "test connection",
                )
                .await;

                if test.response_received {
                    println!(
                        "      Test:   ✓ ({}, {}ms{})",
                        test.layer_resolved.to_uppercase(),
                        test.latency_ms,
                        if test.deflected { ", deflected" } else { "" }
                    );
                } else if test.layer_resolved == "timeout" {
                    println!("      Test:   ~ gateway up, /api/chat timed out (no L3 key?)");
                } else {
                    println!("      Test:   ✗ (no response)");
                }
            }
            None => {
                println!("  ○ {}", client_display_name(client));
                println!("      Not connected. Run: isartor connect {}", client);
            }
        }
    }

    if state.connections.contains_key("claude") && state.connections.contains_key("claude-copilot")
    {
        println!("\n  ⚠ Warning: both Claude Code connectors are active.");
        println!("    Disconnect one to avoid conflicting ~/.claude/settings.json values:");
        println!("      isartor connect claude --disconnect");
        println!("      isartor connect claude-copilot --disconnect");
    }

    let proxy_clients = ["copilot", "claude", "antigravity"];
    if proxy_clients
        .iter()
        .any(|client| state.connections.contains_key(*client))
    {
        println!("\nRecent Proxied Client Requests");
        match check_proxy_recent(
            &gateway,
            effective_gateway_api_key(args.gateway_api_key.as_deref()),
            args.proxy_recent_limit,
        )
        .await
        {
            Some(entries) if !entries.is_empty() => {
                for entry in entries {
                    println!(
                        "  {} {} {} {} via {} ({} ms{})",
                        entry.client,
                        entry.final_layer.to_uppercase(),
                        entry.hostname,
                        entry.path,
                        entry.resolved_by,
                        entry.latency_ms,
                        if entry.deflected { ", deflected" } else { "" }
                    );
                }
            }
            Some(_) => println!("  No proxied client requests recorded yet."),
            None => {
                println!("  Unable to read proxy history. Provide --gateway-api-key if needed.")
            }
        }
    }

    println!();
}

#[derive(Debug, serde::Deserialize)]
struct Health {
    version: String,
    uptime_seconds: u64,
    layers: Layers,
    proxy: String,
    proxy_layer3: String,
    proxy_recent_requests: usize,
}

#[derive(Debug, serde::Deserialize)]
struct Layers {
    l1a: String,
    l1b: String,
    l2: String,
    l3: String,
}

#[derive(Debug, serde::Deserialize)]
struct ProxyRecent {
    entries: Vec<ProxyRouteDecision>,
}

async fn check_isartor_health(gateway: &str) -> Option<Health> {
    let url = format!("{}/health", gateway.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<Health>().await.ok()
}

fn layer_icon(active: bool) -> &'static str {
    if active { "✓" } else { "○" }
}

fn effective_gateway_api_key(cli_value: Option<&str>) -> Option<String> {
    if let Some(value) = cli_value {
        return Some(value.to_string());
    }
    AppConfig::load().ok().map(|cfg| cfg.gateway_api_key)
}

async fn check_proxy_recent(
    gateway: &str,
    gateway_api_key: Option<String>,
    limit: usize,
) -> Option<Vec<ProxyRouteDecision>> {
    let url = format!(
        "{}/debug/proxy/recent?limit={}",
        gateway.trim_end_matches('/'),
        limit
    );
    let client = reqwest::Client::new();
    let mut req = client.get(url).timeout(std::time::Duration::from_secs(2));
    if let Some(key) = gateway_api_key {
        req = req.header("X-API-Key", key);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    Some(resp.json::<ProxyRecent>().await.ok()?.entries)
}

fn client_display_name(client: &str) -> String {
    match client {
        "copilot" => "GitHub Copilot CLI".to_string(),
        "copilot-vscode" => "GitHub Copilot in VS Code".to_string(),
        "claude" => "Claude Code".to_string(),
        "claude-copilot" => "Claude Code + GitHub Copilot".to_string(),
        "openclaw" => "OpenClaw".to_string(),
        "opencode" => "OpenCode".to_string(),
        "antigravity" => "Antigravity".to_string(),
        _ => client.to_string(),
    }
}

fn integration_method(client: &str) -> &'static str {
    match client {
        "copilot" => "MCP server (isartor_chat tool)",
        "copilot-vscode" => "VS Code settings.json debug overrides",
        "claude" => "base URL override (ANTHROPIC_BASE_URL)",
        "claude-copilot" => "base URL override + GitHub Copilot L3 provider",
        "openclaw" => "provider base URL (OpenAI-compatible)",
        "opencode" => "OpenCode provider config (@ai-sdk/openai-compatible)",
        "antigravity" => "base URL override (OpenAI-compatible)",
        _ => "unknown",
    }
}

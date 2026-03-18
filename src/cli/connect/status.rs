use clap::Parser;

use super::{state::ConnectionState, test_isartor_connection, DEFAULT_GATEWAY_URL};

#[derive(Parser, Debug, Clone)]
pub struct StatusArgs {
    /// Isartor gateway URL (default: http://localhost:8080)
    #[arg(long, default_value = DEFAULT_GATEWAY_URL)]
    pub gateway_url: String,

    /// Gateway API key (optional). If omitted, status will still check /health.
    #[arg(long, env = "ISARTOR__GATEWAY_API_KEY")]
    pub gateway_api_key: Option<String>,
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
        }
        None => {
            println!("  URL:     {}", gateway);
            println!("  Status:  ✗ not running");
            println!("  Start:   docker run -p 8080:8080 ghcr.io/isartor-ai/isartor:latest");
        }
    }

    println!("\nConnected Clients");

    let all_clients = ["copilot", "claude", "openclaw", "antigravity"];
    for client in all_clients {
        match state.connections.get(client) {
            Some(conn) => {
                println!("  ✓ {}", client_display_name(client));
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

    println!();
}

#[derive(Debug, serde::Deserialize)]
struct Health {
    version: String,
    uptime_seconds: u64,
    layers: Layers,
}

#[derive(Debug, serde::Deserialize)]
struct Layers {
    l1a: String,
    l1b: String,
    l2: String,
    l3: String,
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
    if active {
        "✓"
    } else {
        "○"
    }
}

fn client_display_name(client: &str) -> String {
    match client {
        "copilot" => "GitHub Copilot CLI".to_string(),
        "claude" => "Claude Code".to_string(),
        "openclaw" => "OpenClaw".to_string(),
        "antigravity" => "Antigravity".to_string(),
        _ => client.to_string(),
    }
}

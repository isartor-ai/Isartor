pub mod antigravity;
pub mod claude;
pub mod claude_copilot;
pub mod codex;
pub mod copilot;
pub mod copilot_vscode;
pub mod cursor;
pub mod gemini;
pub mod generic;
pub mod openclaw;
pub mod opencode;
pub mod state;
pub mod status;

use std::time::Instant;

use anyhow::Context;
use clap::{Parser, Subcommand};

use crate::config::AppConfig;

pub const DEFAULT_GATEWAY_URL: &str = "http://localhost:8080";

#[derive(Parser, Debug, Clone)]
pub struct ConnectArgs {
    #[command(subcommand)]
    pub client: ConnectClient,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ConnectClient {
    /// Connect GitHub Copilot CLI to Isartor
    Copilot(copilot::CopilotArgs),

    /// Connect GitHub Copilot in VS Code to Isartor
    CopilotVscode(copilot_vscode::CopilotVscodeArgs),

    /// Connect Claude Code to Isartor
    Claude(claude::ClaudeArgs),

    /// Connect Claude Code to GitHub Copilot through Isartor
    ClaudeCopilot(claude_copilot::ClaudeCopilotArgs),

    /// Connect OpenClaw to Isartor
    Openclaw(openclaw::OpenclawArgs),

    /// Connect OpenCode to Isartor
    Opencode(opencode::OpencodeArgs),

    /// Connect Antigravity to Isartor
    Antigravity(antigravity::AntigravityArgs),

    /// Connect Cursor IDE to Isartor
    Cursor(cursor::CursorArgs),

    /// Connect OpenAI Codex CLI to Isartor
    Codex(codex::CodexArgs),

    /// Connect Gemini CLI to Isartor
    Gemini(gemini::GeminiArgs),

    /// Connect any OpenAI-compatible tool to Isartor
    Generic(generic::GenericArgs),

    /// Show connection status of all clients
    Status(status::StatusArgs),
}

/// Shared args present on every client command.
#[derive(Parser, Debug, Clone)]
pub struct BaseClientArgs {
    /// Isartor gateway URL (default: http://localhost:8080)
    #[arg(long, default_value = DEFAULT_GATEWAY_URL)]
    pub gateway_url: String,

    /// Gateway API key used to authenticate to Isartor.
    /// Defaults to the locally-loaded AppConfig gateway_api_key when available.
    #[arg(long, env = "ISARTOR__GATEWAY_API_KEY")]
    pub gateway_api_key: Option<String>,

    /// Disconnect this client (restore original config)
    #[arg(long, default_value_t = false)]
    pub disconnect: bool,

    /// Dry-run: show what would change without writing files
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Show the raw config that would be written
    #[arg(long, default_value_t = false)]
    pub show_config: bool,
}

impl BaseClientArgs {
    pub fn effective_gateway_url(&self) -> String {
        // If the user explicitly set it, respect it.
        if self.gateway_url != DEFAULT_GATEWAY_URL {
            return self.gateway_url.clone();
        }

        // Otherwise, try to align with local AppConfig host_port.
        let Ok(cfg) = AppConfig::load_with_validation(false) else {
            return self.gateway_url.clone();
        };

        // host_port is like "0.0.0.0:8080" — map it to localhost.
        let port = cfg
            .host_port
            .rsplit(':')
            .next()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(8080);

        format!("http://localhost:{port}")
    }

    pub fn effective_gateway_api_key(&self) -> Option<String> {
        if self.gateway_api_key.is_some() {
            return self.gateway_api_key.clone();
        }
        AppConfig::load_with_validation(false)
            .ok()
            .map(|c| c.gateway_api_key)
    }
}

/// Result returned by every client connect handler.
#[derive(Debug, Clone)]
pub struct ConnectResult {
    pub client_name: String,
    pub success: bool,
    pub message: String,
    pub changes_made: Vec<ConfigChange>,
    pub test_result: Option<TestResult>,
}

#[derive(Debug, Clone)]
pub struct ConfigChange {
    pub change_type: ConfigChangeType,
    pub target: String, // file path or env var name
    pub description: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ConfigChangeType {
    FileCreated,
    FileModified,
    FileBackedUp,
    EnvVarSet,
    EnvVarRemoved,
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub request_sent: String,
    pub response_received: bool,
    pub layer_resolved: String, // "l1a" | "l1b" | "l2" | "l3" | "l0"
    pub latency_ms: u64,
    pub deflected: bool,
}

pub fn print_connect_result(result: &ConnectResult) {
    let icon = if result.success { "✓" } else { "✗" };
    println!(
        "\n{} Isartor ↔ {} connection {}",
        icon,
        result.client_name,
        if result.success {
            "configured"
        } else {
            "failed"
        }
    );
    println!("{}", result.message);

    if !result.changes_made.is_empty() {
        println!("\nChanges made:");
        for change in &result.changes_made {
            let icon = match change.change_type {
                ConfigChangeType::FileCreated => "  + created: ",
                ConfigChangeType::FileModified => "  ~ updated: ",
                ConfigChangeType::FileBackedUp => "  ↩ backed up: ",
                ConfigChangeType::EnvVarSet => "  $ export:  ",
                ConfigChangeType::EnvVarRemoved => "  $ unset:   ",
            };
            println!("{}{} — {}", icon, change.target, change.description);
        }
    }

    if let Some(test) = &result.test_result {
        println!("\nTest request:");
        if test.response_received {
            println!("  ✓ Response received ({} ms)", test.latency_ms);
            println!(
                "  ✓ Resolved at: {} {}",
                test.layer_resolved.to_uppercase(),
                if test.deflected {
                    "(deflected — $0 cloud cost)"
                } else {
                    "(forwarded to cloud)"
                }
            );
        } else if test.layer_resolved == "timeout" {
            println!(
                "  ~ Gateway reachable, but /api/chat timed out ({} ms)",
                test.latency_ms
            );
            println!("    This is normal when L3 has no API key configured.");
        } else {
            println!("  ✗ No response — is Isartor running?");
        }
    }

    println!();
}

pub async fn test_isartor_connection(
    gateway_url: &str,
    gateway_api_key: Option<&str>,
    test_prompt: &str,
) -> TestResult {
    let base = gateway_url.trim_end_matches('/');
    let client = reqwest::Client::new();

    // First, verify the gateway is reachable via /health (fast, no L3 dependency).
    let health_ok = client
        .get(format!("{base}/health"))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if !health_ok {
        return TestResult {
            request_sent: test_prompt.to_string(),
            response_received: false,
            layer_resolved: "none".to_string(),
            latency_ms: 0,
            deflected: false,
        };
    }

    // Then test the deflection stack. Use a short timeout because L3 may be
    // unconfigured (no API key) and the upstream call can block for 30s+.
    let url = format!("{base}/api/chat");
    let start = Instant::now();

    let mut req = client
        .post(&url)
        .json(&serde_json::json!({ "prompt": test_prompt }));

    if let Some(key) = gateway_api_key {
        req = req.header("X-API-Key", key);
    }

    match req.timeout(std::time::Duration::from_secs(3)).send().await {
        Ok(resp) => {
            let layer = resp
                .headers()
                .get("X-Isartor-Layer")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown")
                .to_string();
            let deflected = resp
                .headers()
                .get("X-Isartor-Deflected")
                .and_then(|v| v.to_str().ok())
                .map(|v| v == "true")
                .unwrap_or(false);

            TestResult {
                request_sent: test_prompt.to_string(),
                response_received: true,
                layer_resolved: layer,
                latency_ms: start.elapsed().as_millis() as u64,
                deflected,
            }
        }
        Err(e) => {
            // Distinguish timeout (gateway running but L3 slow) from connection
            // refused (gateway not running). Health passed so we know it's up.
            let timed_out = e.is_timeout();
            TestResult {
                request_sent: test_prompt.to_string(),
                response_received: false,
                layer_resolved: if timed_out {
                    "timeout".to_string()
                } else {
                    "none".to_string()
                },
                latency_ms: start.elapsed().as_millis() as u64,
                deflected: false,
            }
        }
    }
}

pub async fn handle_connect(args: ConnectArgs) -> anyhow::Result<()> {
    match args.client {
        ConnectClient::Copilot(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = copilot::handle_copilot_connect(a).await;
            print_connect_result(&result);
            update_state("copilot", &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::CopilotVscode(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = copilot_vscode::handle_copilot_vscode_connect(a).await;
            print_connect_result(&result);
            update_state(
                "copilot-vscode",
                &gateway,
                base.disconnect,
                base.dry_run,
                &result,
            );
        }
        ConnectClient::Claude(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = claude::handle_claude_connect(a).await;
            print_connect_result(&result);
            update_state("claude", &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::ClaudeCopilot(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = claude_copilot::handle_claude_copilot_connect(a).await;
            print_connect_result(&result);
            update_state(
                "claude-copilot",
                &gateway,
                base.disconnect,
                base.dry_run,
                &result,
            );
        }
        ConnectClient::Openclaw(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = openclaw::handle_openclaw_connect(a).await;
            print_connect_result(&result);
            update_state("openclaw", &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::Opencode(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = opencode::handle_opencode_connect(a).await;
            print_connect_result(&result);
            update_state("opencode", &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::Antigravity(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = antigravity::handle_antigravity_connect(a).await;
            print_connect_result(&result);
            update_state(
                "antigravity",
                &gateway,
                base.disconnect,
                base.dry_run,
                &result,
            );
        }
        ConnectClient::Cursor(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = cursor::handle_cursor_connect(a).await;
            print_connect_result(&result);
            update_state("cursor", &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::Codex(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = codex::handle_codex_connect(a).await;
            print_connect_result(&result);
            update_state("codex", &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::Gemini(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let result = gemini::handle_gemini_connect(a).await;
            print_connect_result(&result);
            update_state("gemini", &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::Generic(a) => {
            let base = a.base.clone();
            let gateway = base.effective_gateway_url();
            let tool = a.tool_name.clone();
            let result = generic::handle_generic_connect(a).await;
            print_connect_result(&result);
            update_state(&tool, &gateway, base.disconnect, base.dry_run, &result);
        }
        ConnectClient::Status(a) => {
            status::handle_status(a).await;
        }
    }

    Ok(())
}

fn update_state(
    client_id: &str,
    gateway_url: &str,
    disconnect: bool,
    dry_run: bool,
    result: &ConnectResult,
) {
    let mut st = state::ConnectionState::load();

    if disconnect {
        st.connections.remove(client_id);
        st.save();
        return;
    }

    if dry_run || !result.success {
        return;
    }

    let now = chrono::Utc::now().to_rfc3339();

    let mut modified = Vec::new();
    let mut backups = Vec::new();
    for c in &result.changes_made {
        match c.change_type {
            ConfigChangeType::FileCreated | ConfigChangeType::FileModified => {
                modified.push(c.target.clone());
            }
            ConfigChangeType::FileBackedUp => backups.push(c.target.clone()),
            _ => {}
        }
    }

    st.connections.insert(
        client_id.to_string(),
        state::ClientConnection {
            client: client_id.to_string(),
            gateway_url: gateway_url.to_string(),
            connected_at: now,
            config_files_modified: modified,
            backup_files: backups,
        },
    );

    st.save();
}

pub(crate) fn home_path(rel: &str) -> anyhow::Result<std::path::PathBuf> {
    let home = dirs::home_dir().context("could not resolve home directory")?;
    Ok(home.join(rel))
}

pub(crate) fn write_file(
    path: &std::path::Path,
    content: &str,
    dry_run: bool,
) -> anyhow::Result<()> {
    if dry_run {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)?;
    Ok(())
}

pub(crate) fn remove_file(path: &std::path::Path, dry_run: bool) {
    if dry_run {
        return;
    }
    let _ = std::fs::remove_file(path);
}

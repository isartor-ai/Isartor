//! Minimal MCP (Model Context Protocol) stdio server.
//!
//! Run with `isartor mcp` — Copilot CLI (or any MCP client) launches this as
//! a subprocess and communicates via JSON-RPC 2.0 over stdin/stdout.
//!
//! Exposed tools:
//! - `isartor_chat`: Send a prompt through Isartor's deflection stack and
//!   return the response. This enables L1a/L1b cache hits for Copilot.

use std::io::{self, BufRead, Write};

use async_trait::async_trait;
use clap::Parser;
use serde_json::Value;

use super::connect::DEFAULT_GATEWAY_URL;
use crate::mcp::{self, ToolExecutor};

#[derive(Parser, Debug, Clone)]
pub struct McpArgs {
    /// Isartor gateway URL
    #[arg(long, default_value = DEFAULT_GATEWAY_URL, env = "ISARTOR_GATEWAY_URL")]
    pub gateway_url: String,

    /// Gateway API key
    #[arg(long, env = "ISARTOR__GATEWAY_API_KEY")]
    pub gateway_api_key: Option<String>,
}

/// Run the MCP stdio server (blocking — reads stdin line by line).
pub async fn handle_mcp(args: McpArgs) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let executor = GatewayToolExecutor {
        gateway_url: args.gateway_url.clone(),
        gateway_api_key: args.gateway_api_key.clone(),
    };

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue, // ignore non-JSON lines
        };

        let method = msg
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();

        let response = mcp::handle_message(&msg, mcp::STDIO_PROTOCOL_VERSION, &executor).await;

        if let Some(resp) = response {
            send(&mut stdout, &resp)?;
        }

        if method == "shutdown" {
            break;
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct GatewayToolExecutor {
    gateway_url: String,
    gateway_api_key: Option<String>,
}

#[async_trait]
impl ToolExecutor for GatewayToolExecutor {
    async fn cache_lookup(&self, prompt: &str) -> anyhow::Result<Option<String>> {
        mcp::cache_lookup_via_gateway(&self.gateway_url, self.gateway_api_key.as_deref(), prompt)
            .await
    }

    async fn cache_store(&self, prompt: &str, response: &str, model: &str) -> anyhow::Result<()> {
        mcp::cache_store_via_gateway(
            &self.gateway_url,
            self.gateway_api_key.as_deref(),
            prompt,
            response,
            model,
        )
        .await
    }
}

fn send(out: &mut impl Write, msg: &Value) -> io::Result<()> {
    let s = serde_json::to_string(msg).unwrap_or_default();
    writeln!(out, "{s}")?;
    out.flush()
}

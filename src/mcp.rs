use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

pub const STDIO_PROTOCOL_VERSION: &str = "2024-11-05";
pub const STREAMABLE_HTTP_PROTOCOL_VERSION: &str = "2025-03-26";
pub const SESSION_HEADER: &str = "Mcp-Session-Id";

static HTTP_MCP_SESSIONS: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: &'static str,
    pub description: &'static str,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

pub fn initialize_result(protocol_version: &str) -> Value {
    json!({
        "protocolVersion": protocol_version,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "isartor",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "isartor_chat",
            description: "Cache-first lookup for the user's prompt. Call this before answering plain conversational questions. On a hit it returns a cached response from Isartor L1a/L1b; when that happens, return that text verbatim as the final user-facing answer and do not paraphrase or continue reasoning. On a miss it returns empty so Copilot can answer with its own model.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt or question to send"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model name (e.g. gpt-4o-mini)"
                    }
                },
                "required": ["prompt"]
            }),
        },
        ToolDefinition {
            name: "isartor_cache_store",
            description: "Store the final prompt/response pair in Isartor after Copilot answers a cache miss. Only call this after using your own model on a miss; do not call it after a cache hit that already returned a final answer.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The original prompt"
                    },
                    "response": {
                        "type": "string",
                        "description": "The LLM response to cache"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model name"
                    }
                },
                "required": ["prompt", "response"]
            }),
        },
    ]
}

pub fn tools_list_result() -> Value {
    json!({
        "tools": tool_definitions()
    })
}

#[async_trait]
pub trait ToolExecutor {
    async fn cache_lookup(&self, prompt: &str) -> anyhow::Result<Option<String>>;
    async fn cache_store(&self, prompt: &str, response: &str, model: &str) -> anyhow::Result<()>;
}

pub async fn handle_message<E>(msg: &Value, protocol_version: &str, executor: &E) -> Option<Value>
where
    E: ToolExecutor + Sync,
{
    let id = msg.get("id").cloned();
    let method = msg
        .get("method")
        .and_then(|method| method.as_str())
        .unwrap_or_default();

    match method {
        "initialize" => Some(jsonrpc_ok(id?, initialize_result(protocol_version))),
        "notifications/initialized" | "initialized" => None,
        "ping" => Some(jsonrpc_ok(id?, json!({}))),
        "tools/list" => Some(jsonrpc_ok(id?, tools_list_result())),
        "tools/call" => {
            let params = msg.get("params").cloned().unwrap_or(json!({}));
            Some(handle_tools_call(id, &params, executor).await)
        }
        "shutdown" => Some(jsonrpc_ok(id?, json!({}))),
        _ => id.map(|id| jsonrpc_error(id, -32601, "Method not found")),
    }
}

pub fn is_request_message(msg: &Value) -> bool {
    msg.get("method")
        .and_then(|method| method.as_str())
        .is_some()
}

pub fn message_method(msg: &Value) -> Option<&str> {
    msg.get("method").and_then(|method| method.as_str())
}

pub fn register_http_session() -> String {
    let session_id = Uuid::new_v4().to_string();
    HTTP_MCP_SESSIONS
        .lock()
        .expect("HTTP MCP session registry poisoned")
        .insert(session_id.clone());
    session_id
}

pub fn http_session_exists(session_id: &str) -> bool {
    HTTP_MCP_SESSIONS
        .lock()
        .expect("HTTP MCP session registry poisoned")
        .contains(session_id)
}

pub fn remove_http_session(session_id: &str) -> bool {
    HTTP_MCP_SESSIONS
        .lock()
        .expect("HTTP MCP session registry poisoned")
        .remove(session_id)
}

async fn handle_tools_call<E>(id: Option<Value>, params: &Value, executor: &E) -> Value
where
    E: ToolExecutor + Sync,
{
    let id = id.unwrap_or(Value::Null);
    let tool_name = params
        .get("name")
        .and_then(|name| name.as_str())
        .unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    match tool_name {
        "isartor_chat" => {
            let prompt = arguments
                .get("prompt")
                .and_then(|prompt| prompt.as_str())
                .unwrap_or("");

            if prompt.is_empty() {
                return tool_result(id, "Error: prompt is required", true);
            }

            match executor.cache_lookup(prompt).await {
                Ok(Some(answer)) => tool_result(id, &answer, false),
                Ok(None) => tool_result(id, "", false),
                Err(err) => tool_result(id, &format!("Isartor error: {err}"), true),
            }
        }
        "isartor_cache_store" => {
            let prompt = arguments
                .get("prompt")
                .and_then(|prompt| prompt.as_str())
                .unwrap_or("");
            let response = arguments
                .get("response")
                .and_then(|response| response.as_str())
                .unwrap_or("");
            let model = arguments
                .get("model")
                .and_then(|model| model.as_str())
                .unwrap_or("");

            if prompt.is_empty() || response.is_empty() {
                return tool_result(id, "Error: prompt and response are required", true);
            }

            match executor.cache_store(prompt, response, model).await {
                Ok(()) => tool_result(id, "Cached successfully", false),
                Err(err) => tool_result(id, &format!("Cache store error: {err}"), true),
            }
        }
        _ => jsonrpc_error(id, -32602, &format!("Unknown tool: {tool_name}")),
    }
}

fn tool_result(id: Value, text: &str, is_error: bool) -> Value {
    jsonrpc_ok(
        id,
        json!({
            "content": [{
                "type": "text",
                "text": text
            }],
            "isError": is_error
        }),
    )
}

pub fn jsonrpc_ok(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

pub fn jsonrpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message
        }
    })
}

pub async fn cache_lookup_via_gateway(
    gateway_url: &str,
    api_key: Option<&str>,
    prompt: &str,
) -> anyhow::Result<Option<String>> {
    let url = format!("{}/api/v1/cache/lookup", gateway_url.trim_end_matches('/'));
    let client = reqwest::Client::new();

    let mut request = client.post(&url).json(&json!({ "prompt": prompt }));
    if let Some(api_key) = api_key {
        request = request.header("X-API-Key", api_key);
    }

    let response = request
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .context("failed to send cache lookup request")?;

    if response.status() == reqwest::StatusCode::NO_CONTENT {
        return Ok(None);
    }

    if !response.status().is_success() {
        anyhow::bail!("cache lookup failed: {}", response.status());
    }

    let body: Value = response.json().await.unwrap_or(json!({}));
    let answer = body
        .get("message")
        .and_then(|message| message.as_str())
        .or_else(|| body.get("response").and_then(|response| response.as_str()))
        .unwrap_or("")
        .to_string();

    if answer.is_empty() {
        Ok(None)
    } else {
        Ok(Some(answer))
    }
}

pub async fn cache_store_via_gateway(
    gateway_url: &str,
    api_key: Option<&str>,
    prompt: &str,
    response: &str,
    model: &str,
) -> anyhow::Result<()> {
    let url = format!("{}/api/v1/cache/store", gateway_url.trim_end_matches('/'));
    let client = reqwest::Client::new();

    let mut request = client.post(&url).json(&json!({
        "prompt": prompt,
        "response": response,
        "model": model,
    }));
    if let Some(api_key) = api_key {
        request = request.header("X-API-Key", api_key);
    }

    let response = request
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .context("failed to send cache store request")?;

    if !response.status().is_success() {
        anyhow::bail!("cache store failed: {}", response.status());
    }

    Ok(())
}

//! Anthropic Messages API format adapter.

use anyhow::Context;
use axum::response::Response;
use serde_json::Value;

use crate::anthropic_sse;

use super::ApiFormat;
use super::types::{
    InternalContent, InternalMessage, InternalRequest, InternalResponse, InternalRole, InternalTool,
};

pub struct AnthropicFormat;

impl ApiFormat for AnthropicFormat {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn cache_namespace(&self) -> &'static str {
        "anthropic"
    }

    fn parse_request(&self, body: &[u8]) -> anyhow::Result<InternalRequest> {
        parse_anthropic_body(body)
    }

    fn build_response(&self, resp: InternalResponse, streaming: bool) -> Response {
        if streaming {
            anthropic_sse::build_sse_response(&resp.content, &resp.model)
        } else {
            let json = build_anthropic_json(&resp);
            axum::response::IntoResponse::into_response((
                axum::http::StatusCode::OK,
                axum::Json(json),
            ))
        }
    }
}

// ── Parse ─────────────────────────────────────────────────────────────────────

pub fn parse_anthropic_body(body: &[u8]) -> anyhow::Result<InternalRequest> {
    let v: Value = serde_json::from_slice(body).context("invalid JSON")?;

    let model = v
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("claude-3-5-sonnet-20241022")
        .to_owned();
    let stream = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let max_tokens = v
        .get("max_tokens")
        .and_then(|n| n.as_u64())
        .map(|n| n as u32);
    let temperature = v.get("temperature").and_then(|t| t.as_f64());

    // Top-level `system` field (may be string or content-block array)
    let system = parse_anthropic_system(&v);

    let raw_msgs = v
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    let messages: Vec<InternalMessage> = raw_msgs
        .iter()
        .filter_map(|msg| {
            let role_str = msg.get("role")?.as_str()?;
            let role = match role_str {
                "user" => InternalRole::User,
                "assistant" => InternalRole::Assistant,
                _ => InternalRole::User,
            };
            let content = parse_anthropic_content(msg);
            Some(InternalMessage { role, content })
        })
        .collect();

    let tools = parse_anthropic_tools(&v);

    Ok(InternalRequest {
        model,
        system,
        messages,
        tools,
        stream,
        max_tokens,
        temperature,
    })
}

fn parse_anthropic_system(v: &Value) -> Option<String> {
    match v.get("system")? {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => {
            let text: String = blocks
                .iter()
                .filter_map(|b| {
                    if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                        b.get("text")
                            .and_then(|t| t.as_str())
                            .map(ToOwned::to_owned)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() { None } else { Some(text) }
        }
        _ => None,
    }
}

fn parse_anthropic_content(msg: &Value) -> Vec<InternalContent> {
    match msg.get("content") {
        Some(Value::String(s)) => vec![InternalContent::text(s)],
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                let typ = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                match typ {
                    "text" => {
                        let text = block.get("text")?.as_str()?.to_owned();
                        Some(InternalContent::text(text))
                    }
                    "tool_use" => {
                        let id = block.get("id")?.as_str()?.to_owned();
                        let name = block.get("name")?.as_str()?.to_owned();
                        let arguments = block
                            .get("input")
                            .map(|i| i.to_string())
                            .unwrap_or_else(|| "{}".to_string());
                        Some(InternalContent::ToolCall {
                            id,
                            name,
                            arguments,
                        })
                    }
                    "tool_result" => {
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(|id| id.as_str())
                            .unwrap_or("")
                            .to_owned();
                        let content = match block.get("content") {
                            Some(Value::String(s)) => s.clone(),
                            Some(Value::Array(parts)) => parts
                                .iter()
                                .filter_map(|p| {
                                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                                        p.get("text")
                                            .and_then(|t| t.as_str())
                                            .map(ToOwned::to_owned)
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                            _ => String::new(),
                        };
                        Some(InternalContent::ToolResult {
                            tool_use_id,
                            content,
                        })
                    }
                    "image" => {
                        let src = block.get("source")?;
                        let url = if src.get("type").and_then(|t| t.as_str()) == Some("url") {
                            src.get("url")?.as_str()?.to_owned()
                        } else {
                            // base64 — represent as data URL
                            let media_type = src
                                .get("media_type")
                                .and_then(|m| m.as_str())
                                .unwrap_or("image/jpeg");
                            let data = src.get("data")?.as_str()?;
                            format!("data:{media_type};base64,{data}")
                        };
                        Some(InternalContent::ImageUrl { url, detail: None })
                    }
                    _ => None,
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_anthropic_tools(v: &Value) -> Vec<InternalTool> {
    v.get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    let name = tool.get("name")?.as_str()?.to_owned();
                    let description = tool
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(ToOwned::to_owned);
                    let parameters = tool
                        .get("input_schema")
                        .cloned()
                        .unwrap_or(Value::Object(Default::default()));
                    Some(InternalTool {
                        name,
                        description,
                        parameters,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

// ── Build ─────────────────────────────────────────────────────────────────────

pub fn build_anthropic_json(resp: &InternalResponse) -> Value {
    anthropic_sse::build_json_response(&resp.content, &resp.model)
}

// ── InternalRequest → Anthropic wire format ───────────────────────────────────

/// Convert an [`InternalRequest`] into an Anthropic Messages API JSON body.
pub fn internal_to_anthropic_body(req: &InternalRequest) -> Value {
    let messages: Vec<Value> =
        req.messages
            .iter()
            .filter_map(|msg| {
                if msg.role == InternalRole::System {
                    return None;
                }
                let role = match msg.role {
                    InternalRole::Assistant => "assistant",
                    _ => "user",
                };
                let content: Vec<Value> = msg
                .content
                .iter()
                .map(|c| match c {
                    InternalContent::Text { text } => {
                        serde_json::json!({"type": "text", "text": text})
                    }
                    InternalContent::ImageUrl { url, .. } => {
                        serde_json::json!({"type": "image", "source": {"type": "url", "url": url}})
                    }
                    InternalContent::ToolCall { id, name, arguments } => {
                        let input: Value = serde_json::from_str(arguments)
                            .unwrap_or(Value::Object(Default::default()));
                        serde_json::json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input
                        })
                    }
                    InternalContent::ToolResult { tool_use_id, content } => {
                        serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content
                        })
                    }
                })
                .collect();

                Some(serde_json::json!({"role": role, "content": content}))
            })
            .collect();

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": req.stream,
        "max_tokens": req.max_tokens.unwrap_or(4096)
    });

    if let Some(sys) = &req.system {
        body["system"] = Value::String(sys.clone());
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }

    body
}

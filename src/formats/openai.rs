//! OpenAI Chat Completions format adapter.
//!
//! This is the reference implementation — it re-uses the existing
//! [`crate::openai_sse`] SSE builder and the [`crate::models::OpenAiChatRequest`]
//! struct that Isartor has always used internally.

use anyhow::Context;
use axum::response::Response;
use serde_json::Value;

use crate::openai_sse;

use super::ApiFormat;
use super::types::{
    InternalChunk, InternalContent, InternalMessage, InternalRequest, InternalResponse,
    InternalRole, InternalTool,
};

pub struct OpenAiFormat;

impl ApiFormat for OpenAiFormat {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn cache_namespace(&self) -> &'static str {
        "openai"
    }

    fn parse_request(&self, body: &[u8]) -> anyhow::Result<InternalRequest> {
        parse_openai_body(body)
    }

    fn build_response(&self, resp: InternalResponse, streaming: bool) -> Response {
        if streaming {
            openai_sse::build_sse_response(&resp.content, &resp.model)
        } else {
            let json = build_openai_json(&resp);
            axum::response::IntoResponse::into_response((
                axum::http::StatusCode::OK,
                axum::Json(json),
            ))
        }
    }
}

// ── Parse ─────────────────────────────────────────────────────────────────────

pub fn parse_openai_body(body: &[u8]) -> anyhow::Result<InternalRequest> {
    let v: Value = serde_json::from_slice(body).context("invalid JSON")?;
    let model = v
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("gpt-4o")
        .to_owned();
    let stream = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let max_tokens = v
        .get("max_tokens")
        .and_then(|n| n.as_u64())
        .map(|n| n as u32);
    let temperature = v.get("temperature").and_then(|t| t.as_f64());

    let raw_msgs = v
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    let mut system: Option<String> = None;
    let mut messages: Vec<InternalMessage> = Vec::new();

    for msg in &raw_msgs {
        let role_str = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let role = match role_str {
            "system" | "developer" => InternalRole::System,
            "assistant" => InternalRole::Assistant,
            "tool" | "function" => InternalRole::Tool,
            _ => InternalRole::User,
        };

        let content = parse_openai_content(msg);

        if role == InternalRole::System {
            let text: String = content.iter().filter_map(|c| c.as_text()).collect();
            system = Some(text);
        } else {
            messages.push(InternalMessage { role, content });
        }
    }

    // Tool results (role=tool) — keep in messages as ToolResult content
    let tools = parse_openai_tools(&v);

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

fn parse_openai_content(msg: &Value) -> Vec<InternalContent> {
    let role_str = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");

    // tool role → ToolResult
    if role_str == "tool" || role_str == "function" {
        let tool_use_id = msg
            .get("tool_call_id")
            .and_then(|id| id.as_str())
            .unwrap_or("")
            .to_owned();
        let text = msg
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_owned();
        return vec![InternalContent::ToolResult {
            tool_use_id,
            content: text,
        }];
    }

    // assistant role with tool_calls
    if role_str == "assistant"
        && let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array())
    {
        return tool_calls
            .iter()
            .filter_map(|tc| {
                let id = tc.get("id")?.as_str()?.to_owned();
                let fn_obj = tc.get("function")?;
                let name = fn_obj.get("name")?.as_str()?.to_owned();
                let arguments = fn_obj
                    .get("arguments")
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}")
                    .to_owned();
                Some(InternalContent::ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect();
    }

    // standard content: string or array of parts
    match msg.get("content") {
        Some(Value::String(s)) => vec![InternalContent::text(s)],
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                let typ = part.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                match typ {
                    "text" => {
                        let text = part.get("text").and_then(|t| t.as_str())?.to_owned();
                        Some(InternalContent::text(text))
                    }
                    "image_url" => {
                        let url_obj = part.get("image_url")?;
                        let url = url_obj.get("url")?.as_str()?.to_owned();
                        let detail = url_obj
                            .get("detail")
                            .and_then(|d| d.as_str())
                            .map(ToOwned::to_owned);
                        Some(InternalContent::ImageUrl { url, detail })
                    }
                    _ => None,
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_openai_tools(v: &Value) -> Vec<InternalTool> {
    v.get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    let fn_obj = tool.get("function")?;
                    let name = fn_obj.get("name")?.as_str()?.to_owned();
                    let description = fn_obj
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(ToOwned::to_owned);
                    let parameters = fn_obj
                        .get("parameters")
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

pub fn build_openai_json(resp: &InternalResponse) -> Value {
    use std::time::{SystemTime, UNIX_EPOCH};
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let id = format!("chatcmpl-isartor-{}", uuid::Uuid::new_v4().simple());

    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": resp.content},
            "finish_reason": resp.stop_reason.as_deref().unwrap_or("stop")
        }],
        "usage": {
            "prompt_tokens": resp.input_tokens.unwrap_or(0),
            "completion_tokens": resp.output_tokens.unwrap_or(0),
            "total_tokens": resp.input_tokens.unwrap_or(0) + resp.output_tokens.unwrap_or(0)
        }
    })
}

// ── InternalRequest → OpenAI wire format ──────────────────────────────────────

/// Serialise an [`InternalRequest`] into an OpenAI Chat Completions JSON body.
/// Used by the provider-side translation layer when the target provider is
/// OpenAI-compatible but the incoming client was Anthropic/Gemini/etc.
pub fn internal_to_openai_body(req: &InternalRequest) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    // System message first
    if let Some(sys) = &req.system {
        messages.push(serde_json::json!({
            "role": "system",
            "content": sys
        }));
    }

    for msg in &req.messages {
        let role = msg.role.as_openai_str();
        match &msg.content[..] {
            [] => continue,
            [
                InternalContent::ToolResult {
                    tool_use_id,
                    content,
                },
            ] => {
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content
                }));
            }
            parts
                if parts
                    .iter()
                    .all(|c| matches!(c, InternalContent::ToolCall { .. })) =>
            {
                let tool_calls: Vec<Value> = parts
                    .iter()
                    .filter_map(|c| {
                        if let InternalContent::ToolCall {
                            id,
                            name,
                            arguments,
                        } = c
                        {
                            Some(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments}
                            }))
                        } else {
                            None
                        }
                    })
                    .collect();
                messages.push(serde_json::json!({
                    "role": role,
                    "content": null,
                    "tool_calls": tool_calls
                }));
            }
            _ => {
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<Vec<_>>()
                    .join("");
                messages.push(serde_json::json!({"role": role, "content": text}));
            }
        }
    }

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": req.stream
    });

    if let Some(max_tokens) = req.max_tokens {
        body["max_tokens"] = Value::Number(max_tokens.into());
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
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }

    body
}

/// Build a streaming SSE chunk in OpenAI format.
pub fn build_sse_chunk(chunk: &InternalChunk) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let id = format!("chatcmpl-isartor-{}", uuid::Uuid::new_v4().simple());
    let data = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": chunk.model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": chunk.delta},
            "finish_reason": chunk.finish_reason
        }]
    });
    format!("data: {}\n\n", data)
}

//! Gemini GenerateContent format adapter.

use anyhow::Context;
use axum::response::Response;
use serde_json::Value;

use crate::gemini_sse;

use super::ApiFormat;
use super::types::{
    InternalContent, InternalMessage, InternalRequest, InternalResponse, InternalRole, InternalTool,
};

pub struct GeminiFormat;

impl ApiFormat for GeminiFormat {
    fn name(&self) -> &'static str {
        "gemini"
    }

    fn cache_namespace(&self) -> &'static str {
        "gemini"
    }

    fn parse_request(&self, body: &[u8]) -> anyhow::Result<InternalRequest> {
        parse_gemini_body(body)
    }

    fn build_response(&self, resp: InternalResponse, streaming: bool) -> Response {
        if streaming {
            gemini_sse::build_sse_response(&resp.content, &resp.model)
        } else {
            let json = build_gemini_json(&resp);
            axum::response::IntoResponse::into_response((
                axum::http::StatusCode::OK,
                axum::Json(json),
            ))
        }
    }
}

// ── Parse ─────────────────────────────────────────────────────────────────────

pub fn parse_gemini_body(body: &[u8]) -> anyhow::Result<InternalRequest> {
    let v: Value = serde_json::from_slice(body).context("invalid JSON")?;

    // Model comes from the URL path, not the body; use empty string as default.
    let model = v
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_owned();

    let stream = false; // Gemini streaming is determined by the route suffix, not a body field.

    let max_tokens = v
        .get("generationConfig")
        .and_then(|c| c.get("maxOutputTokens"))
        .and_then(|n| n.as_u64())
        .map(|n| n as u32);
    let temperature = v
        .get("generationConfig")
        .and_then(|c| c.get("temperature"))
        .and_then(|t| t.as_f64());

    // systemInstruction
    let system = parse_gemini_system(&v);

    let raw_contents = v
        .get("contents")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    let messages: Vec<InternalMessage> = raw_contents
        .iter()
        .map(|item| {
            let role_str = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let role = match role_str {
                "model" => InternalRole::Assistant,
                _ => InternalRole::User,
            };
            let content = parse_gemini_parts(item);
            InternalMessage { role, content }
        })
        .collect();

    let tools = parse_gemini_tools(&v);

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

fn parse_gemini_system(v: &Value) -> Option<String> {
    let si = v.get("systemInstruction")?;
    let parts = si.get("parts").and_then(|p| p.as_array())?;
    let text: String = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() { None } else { Some(text) }
}

fn parse_gemini_parts(item: &Value) -> Vec<InternalContent> {
    item.get("parts")
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        return Some(InternalContent::text(text));
                    }
                    if let Some(fc) = part.get("functionCall") {
                        let name = fc.get("name")?.as_str()?.to_owned();
                        let arguments = fc
                            .get("args")
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "{}".to_string());
                        let id = name.clone(); // Gemini doesn't have explicit call IDs
                        return Some(InternalContent::ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    if let Some(fr) = part.get("functionResponse") {
                        let name = fr.get("name")?.as_str()?.to_owned();
                        let content = fr
                            .get("response")
                            .map(|r| r.to_string())
                            .unwrap_or_default();
                        return Some(InternalContent::ToolResult {
                            tool_use_id: name,
                            content,
                        });
                    }
                    if let Some(inline_data) = part.get("inlineData") {
                        let media_type = inline_data
                            .get("mimeType")
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/jpeg");
                        let data = inline_data.get("data")?.as_str()?;
                        let url = format!("data:{media_type};base64,{data}");
                        return Some(InternalContent::ImageUrl { url, detail: None });
                    }
                    None
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_gemini_tools(v: &Value) -> Vec<InternalTool> {
    v.get("tools")
        .and_then(|t| t.as_array())
        .map(|tool_groups| {
            tool_groups
                .iter()
                .flat_map(|group| {
                    group
                        .get("functionDeclarations")
                        .and_then(|fd| fd.as_array())
                        .map(|fns| {
                            fns.iter()
                                .filter_map(|f| {
                                    let name = f.get("name")?.as_str()?.to_owned();
                                    let description = f
                                        .get("description")
                                        .and_then(|d| d.as_str())
                                        .map(ToOwned::to_owned);
                                    let parameters = f
                                        .get("parameters")
                                        .cloned()
                                        .unwrap_or(Value::Object(Default::default()));
                                    Some(InternalTool {
                                        name,
                                        description,
                                        parameters,
                                    })
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default()
}

// ── Build ─────────────────────────────────────────────────────────────────────

pub fn build_gemini_json(resp: &InternalResponse) -> Value {
    gemini_sse::build_json_response(&resp.content, &resp.model)
}

// ── InternalRequest → Gemini wire format ─────────────────────────────────────

/// Convert an [`InternalRequest`] into a Gemini `generateContent` JSON body.
pub fn internal_to_gemini_body(req: &InternalRequest) -> Value {
    let contents: Vec<Value> = req
        .messages
        .iter()
        .filter(|m| m.role != InternalRole::System)
        .map(|msg| {
            let role = msg.role.as_gemini_str();
            let parts: Vec<Value> = msg
                .content
                .iter()
                .map(|c| match c {
                    InternalContent::Text { text } => serde_json::json!({"text": text}),
                    InternalContent::ImageUrl { url, .. } => {
                        if url.starts_with("data:") {
                            // Parse data URL to inline data
                            let rest = url.strip_prefix("data:").unwrap_or(url);
                            if let Some((media_type, data)) = rest.split_once(";base64,") {
                                serde_json::json!({
                                    "inlineData": {"mimeType": media_type, "data": data}
                                })
                            } else {
                                serde_json::json!({"text": url})
                            }
                        } else {
                            serde_json::json!({"text": url})
                        }
                    }
                    InternalContent::ToolCall {
                        name, arguments, ..
                    } => {
                        let args: Value = serde_json::from_str(arguments)
                            .unwrap_or(Value::Object(Default::default()));
                        serde_json::json!({
                            "functionCall": {"name": name, "args": args}
                        })
                    }
                    InternalContent::ToolResult {
                        tool_use_id,
                        content,
                    } => {
                        let response: Value = serde_json::from_str(content)
                            .unwrap_or(serde_json::json!({"output": content}));
                        serde_json::json!({
                            "functionResponse": {
                                "name": tool_use_id,
                                "response": response
                            }
                        })
                    }
                })
                .collect();

            serde_json::json!({"role": role, "parts": parts})
        })
        .collect();

    let mut body = serde_json::json!({"contents": contents});

    if let Some(sys) = &req.system {
        body["systemInstruction"] = serde_json::json!({
            "parts": [{"text": sys}]
        });
    }

    let mut gen_config = serde_json::json!({});
    if let Some(max_tokens) = req.max_tokens {
        gen_config["maxOutputTokens"] = Value::Number(max_tokens.into());
    }
    if let Some(temp) = req.temperature {
        gen_config["temperature"] = serde_json::json!(temp);
    }
    if gen_config != serde_json::json!({}) {
        body["generationConfig"] = gen_config;
    }

    if !req.tools.is_empty() {
        let fn_decls: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                })
            })
            .collect();
        body["tools"] = serde_json::json!([{"functionDeclarations": fn_decls}]);
    }

    body
}

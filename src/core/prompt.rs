use serde_json::Value;

/// Extract a stable "prompt string" from various client request formats.
///
/// Supported inputs:
/// - Isartor native: {"prompt": "..."}
/// - OpenAI Chat Completions: {"messages": [{"role": "user", "content": "..."}, ...]}
/// - Anthropic Messages: {"system": "...", "messages": [{"role": "user", "content": "..."|[{"type":"text","text":"..."}, ...]}, ...]}
/// - Gemini GenerateContent: {"contents": [{"role": "user", "parts": [{"text": "..."}]}]}
///
/// Falls back to treating the body as UTF-8.
pub fn extract_prompt(body: &[u8]) -> String {
    extract_prompt_parts(body).0
}

/// Extract a stable cache-key string from various client request formats.
///
/// Unlike `extract_prompt`, this includes OpenAI/Gemini tool definitions and
/// tool-role messages so tool-enabled requests do not collide with plain
/// completions.
pub fn extract_cache_key(body: &[u8]) -> String {
    let (prompt, extras) = extract_prompt_parts(body);
    if extras.is_empty() {
        prompt
    } else if prompt.is_empty() {
        extras.join(
            "
",
        )
    } else {
        format!(
            "{prompt}
{}",
            extras.join(
                "
"
            )
        )
    }
}

pub fn extract_request_model(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(body).ok().and_then(|v| {
        v.get("model")
            .and_then(|m| m.as_str())
            .map(ToOwned::to_owned)
    })
}

pub fn extract_route_model(path: &str) -> Option<String> {
    path.strip_prefix("/v1beta/models/")
        .and_then(|rest| rest.split_once(':').map(|(model, _)| model.to_string()))
}

pub fn cache_namespace_for_path(path: &str) -> &'static str {
    match path {
        "/v1/chat/completions" => "openai",
        "/v1/messages" => "anthropic",
        _ if is_gemini_path(path) => "gemini",
        _ => "native",
    }
}

pub fn is_gemini_path(path: &str) -> bool {
    path.starts_with("/v1beta/models/")
        && (path.contains(":generateContent") || path.contains(":streamGenerateContent"))
}

pub fn is_gemini_streaming_path(path: &str) -> bool {
    path.starts_with("/v1beta/models/") && path.contains(":streamGenerateContent")
}

pub fn override_request_model(body: &[u8], model: &str) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    let Some(object) = value.as_object_mut() else {
        return body.to_vec();
    };
    object.insert("model".to_string(), Value::String(model.to_string()));
    serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec())
}

/// Returns whether the request body includes tool/function fields or tool
/// conversation turns. These requests should not use semantic cache matching.
pub fn has_tooling(body: &[u8]) -> bool {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return false;
    };

    v.get("tools").is_some()
        || v.get("toolConfig").is_some()
        || v.get("tool_config").is_some()
        || v.get("tool_choice").is_some()
        || v.get("functions").is_some()
        || v.get("function_call").is_some()
        || v.get("messages")
            .and_then(|m| m.as_array())
            .map(|messages| messages.iter().any(message_has_tooling))
            .unwrap_or(false)
        || v.get("contents")
            .and_then(|contents| contents.as_array())
            .map(|contents| contents.iter().any(gemini_content_has_tooling))
            .unwrap_or(false)
}

fn extract_prompt_parts(body: &[u8]) -> (String, Vec<String>) {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return (String::from_utf8_lossy(body).to_string(), Vec::new());
    };

    let model_extra = v
        .get("model")
        .and_then(|m| m.as_str())
        .filter(|model| !model.trim().is_empty())
        .map(|model| format!("model: {model}"));

    // 1) Native format: {"prompt": "..."}
    if let Some(p) = v.get("prompt").and_then(|p| p.as_str()) {
        return (p.to_string(), model_extra.into_iter().collect());
    }

    // 2) Gemini native format: {"contents": [...]}.
    if let Some(contents) = v.get("contents").and_then(|c| c.as_array()) {
        let mut parts: Vec<String> = Vec::with_capacity(contents.len() + 1);
        let mut extras: Vec<String> = model_extra.clone().into_iter().collect();

        if let Some(system_instruction) = v
            .get("systemInstruction")
            .or_else(|| v.get("system_instruction"))
        {
            let system_text = extract_gemini_content_text(system_instruction);
            if !system_text.trim().is_empty() {
                parts.push(format!("system: {system_text}"));
            }
        }

        for content in contents {
            let role = content
                .get("role")
                .and_then(|role| role.as_str())
                .unwrap_or("user");
            let rendered = extract_gemini_content_text(content);
            if rendered.trim().is_empty() && !gemini_content_has_tooling(content) {
                continue;
            }

            let mut part = format!("{role}: {rendered}");
            if let Some(parts_value) = content.get("parts")
                && let Some(function_calls) = extract_gemini_function_calls(parts_value)
            {
                part.push_str(&format!(" [function_calls={function_calls}]"));
            }
            parts.push(part);
        }

        if let Some(tools) = v.get("tools") {
            extras.push(format!("tools: {tools}"));
        }
        if let Some(tool_config) = v.get("toolConfig").or_else(|| v.get("tool_config")) {
            extras.push(format!("tool_config: {tool_config}"));
        }

        if !parts.is_empty() {
            return (parts.join("\n"), extras);
        }
    }

    // 3) Chat-like format: {"messages": [...]}.
    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        let mut parts: Vec<String> = Vec::with_capacity(messages.len() + 1);
        let mut extras: Vec<String> = model_extra.clone().into_iter().collect();

        if let Some(system) = v.get("system").and_then(|s| s.as_str())
            && !system.trim().is_empty()
        {
            parts.push(format!("system: {system}"));
        }

        for msg in messages {
            let role = msg
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("unknown");

            let content = extract_message_content(msg);

            if content.trim().is_empty() && role != "tool" && !message_has_tooling(msg) {
                continue;
            }

            let mut part = format!("{role}: {content}");
            if let Some(name) = msg.get("name").and_then(|n| n.as_str()) {
                part.push_str(&format!(" [name={name}]"));
            }
            if let Some(tool_call_id) = msg.get("tool_call_id").and_then(|id| id.as_str()) {
                part.push_str(&format!(" [tool_call_id={tool_call_id}]"));
            }
            if let Some(tool_calls) = msg.get("tool_calls") {
                part.push_str(&format!(" [tool_calls={tool_calls}]"));
            }
            if let Some(function_call) = msg.get("function_call") {
                part.push_str(&format!(" [function_call={function_call}]"));
            }

            parts.push(part);
        }

        if let Some(tools) = v.get("tools") {
            extras.push(format!("tools: {tools}"));
        }
        if let Some(tool_choice) = v.get("tool_choice") {
            extras.push(format!("tool_choice: {tool_choice}"));
        }
        if let Some(functions) = v.get("functions") {
            extras.push(format!("functions: {functions}"));
        }
        if let Some(function_call) = v.get("function_call") {
            extras.push(format!("function_call: {function_call}"));
        }

        if !parts.is_empty() {
            return (parts.join("\n"), extras);
        }
    }

    // 4) Unknown JSON: use the raw JSON string for cache stability.
    (v.to_string(), Vec::new())
}

/// Extract only the **last user message** for semantic (L1b) similarity.
///
/// Multi-turn conversations from Claude Code / Copilot Chat include a large
/// system prompt and full conversation history. When the whole prompt is
/// embedded, the system prompt dominates the vector, causing unrelated
/// questions to appear semantically identical (>0.85 cosine).
///
/// This function returns only the final user turn so the embedding captures
/// the actual question, not the boilerplate context. Falls back to the full
/// prompt when no user message is found.
pub fn extract_semantic_key(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return String::from_utf8_lossy(body).to_string();
    };

    if let Some(p) = v.get("prompt").and_then(|p| p.as_str()) {
        return p.to_string();
    }

    if let Some(contents) = v.get("contents").and_then(|c| c.as_array()) {
        for content in contents.iter().rev() {
            let role = content
                .get("role")
                .and_then(|role| role.as_str())
                .unwrap_or("user");
            if role == "user" {
                let rendered = extract_gemini_content_text(content);
                if !rendered.trim().is_empty() {
                    return rendered;
                }
            }
        }
    }

    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        for msg in messages.iter().rev() {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role == "user" {
                let content = extract_message_content(msg);
                if !content.trim().is_empty() {
                    return content;
                }
            }
        }
    }

    extract_prompt(body)
}

fn extract_gemini_content_text(content: &Value) -> String {
    let Some(parts) = content.get("parts").and_then(|parts| parts.as_array()) else {
        return content
            .get("text")
            .and_then(|text| text.as_str())
            .unwrap_or_default()
            .to_string();
    };

    let mut rendered_parts = Vec::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(|text| text.as_str()) {
            if !text.trim().is_empty() {
                rendered_parts.push(text.to_string());
            }
            continue;
        }

        if let Some(function_call) = part
            .get("functionCall")
            .or_else(|| part.get("function_call"))
        {
            rendered_parts.push(format!("[function_call={function_call}]"));
            continue;
        }

        if !part.is_null() {
            rendered_parts.push(part.to_string());
        }
    }

    rendered_parts.join("\n")
}

/// Extract the text content from a single message object.
fn extract_message_content(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(Value::Array(blocks)) => {
            let mut buf = String::new();
            for block in blocks {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(text);
                }
            }
            buf
        }
        Some(other) => other.to_string(),
    }
}

fn message_has_tooling(msg: &Value) -> bool {
    msg.get("role").and_then(|r| r.as_str()) == Some("tool")
        || msg.get("tool_call_id").is_some()
        || msg.get("tool_calls").is_some()
        || msg.get("function_call").is_some()
}

fn gemini_content_has_tooling(content: &Value) -> bool {
    content
        .get("parts")
        .and_then(|parts| parts.as_array())
        .map(|parts| {
            parts.iter().any(|part| {
                part.get("functionCall").is_some()
                    || part.get("function_call").is_some()
                    || part.get("functionResponse").is_some()
                    || part.get("function_response").is_some()
                    || part.get("executableCode").is_some()
                    || part.get("codeExecutionResult").is_some()
            })
        })
        .unwrap_or(false)
}

fn extract_gemini_function_calls(parts: &Value) -> Option<String> {
    let parts = parts.as_array()?;
    let calls = parts
        .iter()
        .filter_map(|part| {
            part.get("functionCall")
                .or_else(|| part.get("function_call"))
        })
        .cloned()
        .collect::<Vec<_>>();
    if calls.is_empty() {
        None
    } else {
        Some(Value::Array(calls).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_native_prompt() {
        let body = br#"{"prompt":"hello"}"#;
        assert_eq!(extract_prompt(body), "hello");
    }

    #[test]
    fn extracts_openai_messages() {
        let body = br#"{"model":"gpt","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"2+2?"}]}"#;
        let p = extract_prompt(body);
        assert!(p.contains("system: be brief"));
        assert!(p.contains("user: 2+2?"));
    }

    #[test]
    fn extracts_anthropic_blocks() {
        let body = br#"{"system":"hi","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}"#;
        let p = extract_prompt(body);
        assert!(p.contains("system: hi"));
        assert!(p.contains("user: hello"));
    }

    #[test]
    fn extracts_gemini_contents_and_system_instruction() {
        let body = br#"{
            "model":"gemini-2.0-flash",
            "systemInstruction":{"parts":[{"text":"be brief"}]},
            "contents":[{"role":"user","parts":[{"text":"hello gemini"}]}]
        }"#;
        let prompt = extract_prompt(body);
        assert!(prompt.contains("system: be brief"));
        assert!(prompt.contains("user: hello gemini"));
    }

    #[test]
    fn semantic_key_returns_last_user_message_from_multi_turn() {
        let body = br#"{"system":"You are a helpful assistant","messages":[
            {"role":"user","content":"What is 2+2?"},
            {"role":"assistant","content":"4"},
            {"role":"user","content":"What is the capital of France?"}
        ]}"#;
        let key = extract_semantic_key(body);
        assert_eq!(key, "What is the capital of France?");
    }

    #[test]
    fn semantic_key_returns_last_user_from_anthropic_blocks() {
        let body = br#"{"system":"hi","messages":[
            {"role":"user","content":[{"type":"text","text":"explain Rust"}]}
        ]}"#;
        let key = extract_semantic_key(body);
        assert_eq!(key, "explain Rust");
    }

    #[test]
    fn semantic_key_returns_prompt_for_native_format() {
        let body = br#"{"prompt":"hello world"}"#;
        assert_eq!(extract_semantic_key(body), "hello world");
    }

    #[test]
    fn semantic_key_returns_last_gemini_user_turn() {
        let body = br#"{
            "contents":[
                {"role":"user","parts":[{"text":"first"}]},
                {"role":"model","parts":[{"text":"answer"}]},
                {"role":"user","parts":[{"text":"second"}]}
            ]
        }"#;
        assert_eq!(extract_semantic_key(body), "second");
    }

    #[test]
    fn semantic_key_ignores_system_prompt() {
        let body = br#"{"system":"You are Claude, an AI assistant made by Anthropic. You are extremely helpful, harmless, and honest. You have extensive knowledge about programming, science, math, and many other topics.","messages":[
            {"role":"user","content":"What is 1+1?"}
        ]}"#;
        assert_eq!(extract_semantic_key(body), "What is 1+1?");
    }

    #[test]
    fn semantic_key_different_questions_are_different() {
        let body1 = br#"{"system":"be helpful","messages":[{"role":"user","content":"capital of France"}]}"#;
        let body2 = br#"{"system":"be helpful","messages":[{"role":"user","content":"capital of Germany"}]}"#;
        let k1 = extract_semantic_key(body1);
        let k2 = extract_semantic_key(body2);
        assert_ne!(k1, k2);
        assert_eq!(k1, "capital of France");
        assert_eq!(k2, "capital of Germany");
    }

    #[test]
    fn cache_key_includes_model_identifier() {
        let body = br#"{"model":"fast","messages":[{"role":"user","content":"hello"}]}"#;
        let key = extract_cache_key(body);
        assert!(key.contains("model: fast"));
        assert!(key.contains("user: hello"));
    }

    #[test]
    fn cache_key_includes_top_level_tool_fields() {
        let body = br#"{
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"weather?"}],
            "tools":[{"type":"function","function":{"name":"lookup_weather"}}],
            "tool_choice":{"type":"function","function":{"name":"lookup_weather"}},
            "functions":[{"name":"legacy_lookup"}]
        }"#;
        let key = extract_cache_key(body);
        assert!(key.contains("user: weather?"));
        assert!(key.contains("tools:"));
        assert!(key.contains("tool_choice:"));
        assert!(key.contains("functions:"));
    }

    #[test]
    fn cache_key_includes_tool_role_history() {
        let body = br##"{
            "model":"gpt-4o",
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"call_1","name":"lookup","content":"{\"ok\":true}"}
            ]
        }"##;
        let key = extract_cache_key(body);
        assert!(key.contains("[tool_calls="));
        assert!(key.contains("tool: "));
        assert!(key.contains("[name=lookup]"));
        assert!(key.contains("[tool_call_id=call_1]"));
    }

    #[test]
    fn semantic_detection_marks_tooling_requests() {
        let body = br##"{
            "model":"gpt-4o",
            "messages":[{"role":"tool","tool_call_id":"call_1","content":"{\"ok\":true}"}]
        }"##;
        assert!(has_tooling(body));
        assert!(!has_tooling(
            br#"{"messages":[{"role":"user","content":"hello"}]}"#
        ));
    }

    #[test]
    fn gemini_tooling_is_detected() {
        let body = br#"{
            "tools":[{"functionDeclarations":[{"name":"lookup"}]}],
            "contents":[{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"city":"Berlin"}}}]}]
        }"#;
        assert!(has_tooling(body));
    }

    #[test]
    fn request_model_can_be_extracted_and_overridden() {
        let body = br#"{"prompt":"hello","model":"fast"}"#;
        assert_eq!(extract_request_model(body).as_deref(), Some("fast"));

        let overridden = override_request_model(body, "gpt-4o-mini");
        assert_eq!(
            extract_request_model(&overridden).as_deref(),
            Some("gpt-4o-mini")
        );
    }

    #[test]
    fn extracts_gemini_route_model() {
        assert_eq!(
            extract_route_model("/v1beta/models/gemini-2.0-flash:generateContent").as_deref(),
            Some("gemini-2.0-flash")
        );
        assert!(is_gemini_path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        ));
        assert!(is_gemini_streaming_path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        ));
        assert_eq!(
            cache_namespace_for_path("/v1beta/models/gemini-2.0-flash:generateContent"),
            "gemini"
        );
    }
}

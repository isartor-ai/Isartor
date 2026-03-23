use serde_json::Value;

/// Extract a stable "prompt string" from various client request formats.
///
/// Supported inputs:
/// - Isartor native: {"prompt": "..."}
/// - OpenAI Chat Completions: {"messages": [{"role": "user", "content": "..."}, ...]}
/// - Anthropic Messages: {"system": "...", "messages": [{"role": "user", "content": "..."|[{"type":"text","text":"..."}, ...]}, ...]}
///
/// Falls back to treating the body as UTF-8.
pub fn extract_prompt(body: &[u8]) -> String {
    extract_prompt_parts(body).0
}

/// Extract a stable cache-key string from various client request formats.
///
/// Unlike `extract_prompt`, this includes OpenAI tool definitions and tool-role
/// messages so tool-enabled requests do not collide with plain completions.
pub fn extract_cache_key(body: &[u8]) -> String {
    let (prompt, extras) = extract_prompt_parts(body);
    if extras.is_empty() {
        prompt
    } else if prompt.is_empty() {
        extras.join("\n")
    } else {
        format!("{prompt}\n{}", extras.join("\n"))
    }
}

/// Returns whether the request body includes OpenAI tool/function fields or tool
/// conversation turns. These requests should not use semantic cache matching.
pub fn has_tooling(body: &[u8]) -> bool {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return false;
    };

    v.get("tools").is_some()
        || v.get("tool_choice").is_some()
        || v.get("functions").is_some()
        || v.get("function_call").is_some()
        || v.get("messages")
            .and_then(|m| m.as_array())
            .map(|messages| messages.iter().any(message_has_tooling))
            .unwrap_or(false)
}

fn extract_prompt_parts(body: &[u8]) -> (String, Vec<String>) {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return (String::from_utf8_lossy(body).to_string(), Vec::new());
    };

    // 1) Native format: {"prompt": "..."}
    if let Some(p) = v.get("prompt").and_then(|p| p.as_str()) {
        return (p.to_string(), Vec::new());
    }

    // 2) Chat-like format: {"messages": [...]}
    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        let mut parts: Vec<String> = Vec::with_capacity(messages.len() + 1);
        let mut extras: Vec<String> = Vec::new();

        // Anthropic supports a top-level system field.
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

            // Skip empty messages to avoid creating accidental identical prompts.
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
                part.push_str(&format!(" [tool_calls={}]", tool_calls));
            }
            if let Some(function_call) = msg.get("function_call") {
                part.push_str(&format!(" [function_call={}]", function_call));
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

    // 3) Unknown JSON: use the raw JSON string for cache stability.
    (v.to_string(), Vec::new())
}

/// Extract only the **last user message** for semantic (L1b) similarity.
///
/// Multi-turn conversations from Claude Code / Copilot Chat include a large
/// system prompt and full conversation history.  When the whole prompt is
/// embedded, the system prompt dominates the vector, causing unrelated
/// questions to appear semantically identical (>0.85 cosine).
///
/// This function returns only the final user turn so the embedding captures
/// the actual question, not the boilerplate context.  Falls back to the full
/// prompt when no user message is found.
pub fn extract_semantic_key(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return String::from_utf8_lossy(body).to_string();
    };

    // Native format: just return the prompt as-is.
    if let Some(p) = v.get("prompt").and_then(|p| p.as_str()) {
        return p.to_string();
    }

    // Chat-like: find the last user message.
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

    // Fallback: full extraction.
    extract_prompt(body)
}

/// Extract the text content from a single message object.
fn extract_message_content(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        // Anthropic: content is an array of blocks.
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

    // -- extract_semantic_key tests --

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
    fn semantic_key_ignores_system_prompt() {
        // The system prompt is huge but the question is short.
        // Semantic key should return only the question.
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
        let body = br#"{
            "model":"gpt-4o",
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"call_1","name":"lookup","content":"{\"ok\":true}"}
            ]
        }"#;
        let key = extract_cache_key(body);
        assert!(key.contains("[tool_calls="));
        assert!(key.contains("tool: "));
        assert!(key.contains("[name=lookup]"));
        assert!(key.contains("[tool_call_id=call_1]"));
    }

    #[test]
    fn semantic_detection_marks_tooling_requests() {
        let body = br#"{
            "model":"gpt-4o",
            "messages":[{"role":"tool","tool_call_id":"call_1","content":"{\"ok\":true}"}]
        }"#;
        assert!(has_tooling(body));
        assert!(!has_tooling(
            br#"{"messages":[{"role":"user","content":"hello"}]}"#
        ));
    }
}

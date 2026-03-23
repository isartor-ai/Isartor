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
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return String::from_utf8_lossy(body).to_string();
    };

    // 1) Native format: {"prompt": "..."}
    if let Some(p) = v.get("prompt").and_then(|p| p.as_str()) {
        return p.to_string();
    }

    // 2) Chat-like format: {"messages": [...]}
    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        let mut parts: Vec<String> = Vec::with_capacity(messages.len() + 1);

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
            if content.trim().is_empty() {
                continue;
            }

            parts.push(format!("{role}: {content}"));
        }

        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    // 3) Unknown JSON: use the raw JSON string for cache stability.
    v.to_string()
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
        None => String::new(),
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
}

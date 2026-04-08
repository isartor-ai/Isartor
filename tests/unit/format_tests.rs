// Format translation matrix unit tests.
//
// Covers: parse + round-trip for all 5 client formats, provider-side
// translation (OpenAI ↔ Anthropic ↔ Gemini), cache namespace isolation,
// and Cursor/Kiro header detection.

use axum::http::HeaderMap;
use isartor::config::LlmProvider;
use isartor::formats::translate::ProviderWireFormat;
use isartor::formats::{
    self,
    anthropic::{internal_to_anthropic_body, parse_anthropic_body},
    gemini::{internal_to_gemini_body, parse_gemini_body},
    openai::{internal_to_openai_body, parse_openai_body},
    translate::provider_wire_format,
    types::{InternalContent, InternalRole},
};

// ── OpenAI parse ──────────────────────────────────────────────────────

#[test]
fn openai_parse_simple_user_message() {
    let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}"#;
    let req = parse_openai_body(body).unwrap();
    assert_eq!(req.model, "gpt-4o");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, InternalRole::User);
    assert_eq!(req.messages[0].text_content(), "Hello");
    assert!(req.system.is_none());
}

#[test]
fn openai_parse_system_message_extracted() {
    let body = br#"{
            "model":"gpt-4o",
            "messages":[
                {"role":"system","content":"You are helpful."},
                {"role":"user","content":"Hi"}
            ]
        }"#;
    let req = parse_openai_body(body).unwrap();
    assert_eq!(req.system.as_deref(), Some("You are helpful."));
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, InternalRole::User);
}

#[test]
fn openai_parse_streaming_flag() {
    let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"X"}],"stream":true}"#;
    let req = parse_openai_body(body).unwrap();
    assert!(req.stream);
}

#[test]
fn openai_parse_tool_call_assistant() {
    let body = br#"{
            "model": "gpt-4o",
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}
                }]
            }]
        }"#;
    let req = parse_openai_body(body).unwrap();
    assert_eq!(req.messages.len(), 1);
    let content = &req.messages[0].content[0];
    assert!(
        matches!(content, InternalContent::ToolCall { id, name, .. } if id == "call_abc" && name == "get_weather")
    );
}

#[test]
fn openai_parse_tool_result() {
    let body = br#"{
            "model": "gpt-4o",
            "messages": [{
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "It's sunny in London."
            }]
        }"#;
    let req = parse_openai_body(body).unwrap();
    let content = &req.messages[0].content[0];
    assert!(
        matches!(content, InternalContent::ToolResult { tool_use_id, content } if tool_use_id == "call_abc" && content == "It's sunny in London.")
    );
}

// ── Anthropic parse ───────────────────────────────────────────────────

#[test]
fn anthropic_parse_simple() {
    let body = br#"{
            "model": "claude-3-5-sonnet-20241022",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024
        }"#;
    let req = parse_anthropic_body(body).unwrap();
    assert_eq!(req.model, "claude-3-5-sonnet-20241022");
    assert_eq!(req.system.as_deref(), Some("Be concise."));
    assert_eq!(req.messages[0].text_content(), "Hello");
    assert_eq!(req.max_tokens, Some(1024));
}

#[test]
fn anthropic_parse_content_block_array() {
    let body = br#"{
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{
                "role": "user",
                "content": [{"type":"text","text":"Hello world"}]
            }],
            "max_tokens": 512
        }"#;
    let req = parse_anthropic_body(body).unwrap();
    assert_eq!(req.messages[0].text_content(), "Hello world");
}

#[test]
fn anthropic_parse_tool_use_block() {
    let body = br#"{
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search",
                    "input": {"query": "Rust language"}
                }]
            }],
            "max_tokens": 512
        }"#;
    let req = parse_anthropic_body(body).unwrap();
    let content = &req.messages[0].content[0];
    assert!(
        matches!(content, InternalContent::ToolCall { id, name, .. } if id == "toolu_01" && name == "search")
    );
}

// ── Gemini parse ──────────────────────────────────────────────────────

#[test]
fn gemini_parse_simple() {
    let body = br#"{
            "contents": [{"role": "user", "parts": [{"text": "What is Rust?"}]}]
        }"#;
    let req = parse_gemini_body(body).unwrap();
    assert_eq!(req.messages[0].text_content(), "What is Rust?");
    assert_eq!(req.messages[0].role, InternalRole::User);
}

#[test]
fn gemini_parse_model_role() {
    let body = br#"{
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello!"}]}
            ]
        }"#;
    let req = parse_gemini_body(body).unwrap();
    assert_eq!(req.messages[1].role, InternalRole::Assistant);
}

#[test]
fn gemini_parse_system_instruction() {
    let body = br#"{
            "systemInstruction": {"parts": [{"text": "Be a pirate."}]},
            "contents": [{"role": "user", "parts": [{"text": "Ahoy"}]}]
        }"#;
    let req = parse_gemini_body(body).unwrap();
    assert_eq!(req.system.as_deref(), Some("Be a pirate."));
}

// ── Round-trip: OpenAI → internal → OpenAI ────────────────────────────

#[test]
fn openai_round_trip_preserves_messages() {
    let req = parse_openai_body(
        br#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"}
            ]
        }"#,
    )
    .unwrap();

    let body = internal_to_openai_body(&req);
    let msgs = body["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "Be helpful.");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "What is 2+2?");
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["content"], "4");
}

// ── Cross-format: Anthropic → OpenAI ─────────────────────────────────

#[test]
fn anthropic_to_openai_translation() {
    let req = parse_anthropic_body(
        br#"{
            "model": "claude-3-5-sonnet-20241022",
            "system": "Be precise.",
            "messages": [{"role": "user", "content": "Explain Rust lifetimes."}],
            "max_tokens": 512
        }"#,
    )
    .unwrap();

    let body = internal_to_openai_body(&req);
    let msgs = body["messages"].as_array().unwrap();
    // System should appear as first OpenAI message with role=system
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "Be precise.");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "Explain Rust lifetimes.");
    assert_eq!(body["max_tokens"], 512);
}

// ── Cross-format: Gemini → Anthropic ─────────────────────────────────

#[test]
fn gemini_to_anthropic_translation() {
    let req = parse_gemini_body(
        br#"{
            "systemInstruction": {"parts": [{"text": "You are helpful."}]},
            "contents": [{"role": "user", "parts": [{"text": "Hi there"}]}]
        }"#,
    )
    .unwrap();

    let body = internal_to_anthropic_body(&req);
    assert_eq!(body["system"], "You are helpful.");
    let msgs = body["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "user");
}

// ── Cross-format: OpenAI → Gemini ─────────────────────────────────────

#[test]
fn openai_to_gemini_translation() {
    let req = parse_openai_body(
        br#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"}
            ]
        }"#,
    )
    .unwrap();

    let body = internal_to_gemini_body(&req);
    assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be concise.");
    let contents = body["contents"].as_array().unwrap();
    assert_eq!(contents[0]["role"], "user");
    assert_eq!(contents[0]["parts"][0]["text"], "Hello");
}

// ── Provider wire format selection ────────────────────────────────────

#[test]
fn provider_wire_format_openai_family() {
    assert_eq!(
        provider_wire_format(&LlmProvider::Openai),
        ProviderWireFormat::OpenAi
    );
    assert_eq!(
        provider_wire_format(&LlmProvider::Groq),
        ProviderWireFormat::OpenAi
    );
    assert_eq!(
        provider_wire_format(&LlmProvider::Azure),
        ProviderWireFormat::OpenAi
    );
    assert_eq!(
        provider_wire_format(&LlmProvider::Copilot),
        ProviderWireFormat::OpenAi
    );
}

#[test]
fn provider_wire_format_anthropic() {
    assert_eq!(
        provider_wire_format(&LlmProvider::Anthropic),
        ProviderWireFormat::Anthropic
    );
}

#[test]
fn provider_wire_format_gemini() {
    assert_eq!(
        provider_wire_format(&LlmProvider::Gemini),
        ProviderWireFormat::Gemini
    );
}

// ── Cache namespace isolation ─────────────────────────────────────────

#[test]
fn cache_namespace_by_path() {
    let empty = HeaderMap::new();
    assert_eq!(
        formats::cache_namespace("/v1/chat/completions", &empty),
        "openai"
    );
    assert_eq!(
        formats::cache_namespace("/v1/messages", &empty),
        "anthropic"
    );
    assert_eq!(
        formats::cache_namespace("/v1beta/models/gemini-2.0-flash:generateContent", &empty),
        "gemini"
    );
    assert_eq!(formats::cache_namespace("/api/chat", &empty), "native");
}

#[test]
fn cache_namespace_cursor_from_headers() {
    let mut headers = HeaderMap::new();
    headers.insert("x-cursor-checksum", "abc123".parse().unwrap());
    assert_eq!(
        formats::cache_namespace("/v1/chat/completions", &headers),
        "cursor"
    );
}

#[test]
fn cache_namespace_kiro_from_headers() {
    let mut headers = HeaderMap::new();
    headers.insert("x-kiro-version", "1.0".parse().unwrap());
    assert_eq!(
        formats::cache_namespace("/v1/chat/completions", &headers),
        "kiro"
    );
}

// ── detect_format returns correct trait objects ────────────────────────

#[test]
fn detect_format_anthropic() {
    let headers = HeaderMap::new();
    let fmt = formats::detect_format("/v1/messages", &headers);
    assert_eq!(fmt.cache_namespace(), "anthropic");
    assert_eq!(fmt.name(), "anthropic");
}

#[test]
fn detect_format_cursor_headers_override_path() {
    let mut headers = HeaderMap::new();
    headers.insert("x-cursor-client-version", "0.43.0".parse().unwrap());
    let fmt = formats::detect_format("/v1/chat/completions", &headers);
    assert_eq!(fmt.cache_namespace(), "cursor");
}

// ── InternalRequest prompt extraction ─────────────────────────────────

#[test]
fn internal_request_to_prompt_includes_system_and_user() {
    let req = parse_openai_body(
        br#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What is Rust?"}
            ]
        }"#,
    )
    .unwrap();
    let prompt = req.to_prompt();
    assert!(prompt.contains("[system] Be helpful."));
    assert!(prompt.contains("[user] What is Rust?"));
}

#[test]
fn internal_request_last_user_text() {
    let req = parse_openai_body(
        br#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Third"}
            ]
        }"#,
    )
    .unwrap();
    assert_eq!(req.last_user_text().as_deref(), Some("Third"));
}

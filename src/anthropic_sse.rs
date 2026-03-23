//! Anthropic Messages API Server-Sent Events (SSE) helpers.
//!
//! Claude Code and other Anthropic-compatible clients request `"stream": true`,
//! expecting the response as a series of SSE events rather than a single JSON
//! body. This module converts a plain-text LLM response into the correct
//! Anthropic streaming wire format.

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use uuid::Uuid;

/// Returns `true` if the request body JSON contains `"stream": true`.
pub fn is_streaming_request(body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream")?.as_bool())
        .unwrap_or(false)
}

/// Build a complete Anthropic Messages SSE response from a plain-text answer.
///
/// The response includes all required SSE events that Claude Code expects:
/// `message_start`, `content_block_start`, `ping`, `content_block_delta`,
/// `content_block_stop`, `message_delta`, `message_stop`.
pub fn build_sse_response(text: &str, model: &str) -> Response {
    let msg_id = format!("msg_isartor_{}", Uuid::new_v4().simple());
    let output_tokens = estimate_tokens(text);

    let events = format!(
        "event: message_start\n\
         data: {}\n\n\
         event: content_block_start\n\
         data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
         event: ping\n\
         data: {{\"type\":\"ping\"}}\n\n\
         event: content_block_delta\n\
         data: {}\n\n\
         event: content_block_stop\n\
         data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
         event: message_delta\n\
         data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":{output_tokens}}}}}\n\n\
         event: message_stop\n\
         data: {{\"type\":\"message_stop\"}}\n\n",
        serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": serde_json::Value::Null,
                "stop_sequence": serde_json::Value::Null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }),
        serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text}
        }),
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(Bytes::from(events)))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Build a standard (non-streaming) Anthropic Messages JSON response with all
/// required fields that Claude Code expects.
pub fn build_json_response(text: &str, model: &str) -> serde_json::Value {
    let msg_id = format!("msg_isartor_{}", Uuid::new_v4().simple());
    let output_tokens = estimate_tokens(text);

    serde_json::json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": serde_json::Value::Null,
        "usage": {"input_tokens": 0, "output_tokens": output_tokens}
    })
}

/// Build an SSE response from a cached JSON string (for cache hit paths).
///
/// Parses the cached body to extract the text, then wraps it in SSE events.
/// Falls back to returning the raw cached body if parsing fails.
pub fn cached_to_sse_response(cached_json: &str, model_fallback: &str) -> Response {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(cached_json) {
        // Try to extract text from Anthropic format: content[0].text
        let text = val
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|block| block.get("text"))
            .and_then(|t| t.as_str())
            // Fallback: native ChatResponse format
            .or_else(|| val.get("message").and_then(|m| m.as_str()));

        let model = val
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or(model_fallback);

        if let Some(text) = text {
            return build_sse_response(text, model);
        }
    }

    // Parsing failed — return the raw cached JSON as-is.
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(cached_json.to_owned()))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Rough token estimate (1 token ≈ 4 chars). Good enough for the `usage` field.
fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_streaming_request() {
        let body = br#"{"model":"claude-sonnet-4.5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        assert!(is_streaming_request(body));
    }

    #[test]
    fn detects_non_streaming_request() {
        let body = br#"{"model":"claude-sonnet-4.5","messages":[{"role":"user","content":"hi"}]}"#;
        assert!(!is_streaming_request(body));

        let body2 = br#"{"model":"claude-sonnet-4.5","stream":false,"messages":[]}"#;
        assert!(!is_streaming_request(body2));
    }

    #[test]
    fn build_sse_contains_all_events() {
        let resp = build_sse_response("Hello world", "claude-sonnet-4.5");
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
    }

    #[test]
    fn build_json_has_required_fields() {
        let val = build_json_response("test", "model-1");
        assert!(val.get("id").is_some());
        assert_eq!(val["type"], "message");
        assert_eq!(val["role"], "assistant");
        assert_eq!(val["stop_reason"], "end_turn");
        assert!(val.get("usage").is_some());
    }

    #[test]
    fn cached_anthropic_json_converts_to_sse() {
        let cached = r#"{"type":"message","role":"assistant","model":"gpt-4o","content":[{"type":"text","text":"cached answer"}],"stop_reason":"end_turn"}"#;
        let resp = cached_to_sse_response(cached, "fallback-model");
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
    }

    #[test]
    fn cached_native_json_converts_to_sse() {
        let cached = r#"{"layer":1,"message":"cached native","model":"gpt-4o"}"#;
        let resp = cached_to_sse_response(cached, "fallback");
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
    }
}

//! OpenAI Chat Completions Server-Sent Events (SSE) helpers.
//!
//! OpenAI-compatible clients send `"stream": true` and expect a sequence of
//! `data:` events containing `chat.completion.chunk` payloads, terminated by
//! `data: [DONE]`. This module converts canonical JSON responses into that wire
//! format at the HTTP boundary so caches can continue storing plain JSON.

use std::time::{SystemTime, UNIX_EPOCH};

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

/// Build a complete OpenAI-compatible SSE response from a plain-text answer.
pub fn build_sse_response(text: &str, model: &str) -> Response {
    let completion_id = format!("chatcmpl-isartor-{}", Uuid::new_v4().simple());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let events = format!(
        "data: {}\n\n\
         data: {}\n\n\
         data: {}\n\n\
         data: [DONE]\n\n",
        serde_json::json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": serde_json::Value::Null
            }]
        }),
        serde_json::json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": serde_json::Value::Null
            }]
        }),
        serde_json::json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }),
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(Bytes::from(events)))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn build_tool_call_sse_response(
    tool_calls: &serde_json::Value,
    finish_reason: &str,
    model: &str,
) -> Response {
    let completion_id = format!("chatcmpl-isartor-{}", Uuid::new_v4().simple());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let events = format!(
        "data: {}\n\n\
         data: {}\n\n\
         data: {}\n\n\
         data: [DONE]\n\n",
        serde_json::json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": serde_json::Value::Null
            }]
        }),
        serde_json::json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": tool_calls},
                "finish_reason": serde_json::Value::Null
            }]
        }),
        serde_json::json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        }),
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(Bytes::from(events)))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Build an SSE response from a cached JSON string (for cache hit paths).
pub fn cached_to_sse_response(cached_json: &str, model_fallback: &str) -> Response {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(cached_json) {
        let choice = val
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first());

        let text = choice
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|content| content.as_str())
            .or_else(|| val.get("message").and_then(|message| message.as_str()));

        let tool_calls = choice
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("tool_calls"));

        let function_call = choice
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("function_call"));

        let model = val
            .get("model")
            .and_then(|model| model.as_str())
            .unwrap_or(model_fallback);

        if let Some(text) = text {
            return build_sse_response(text, model);
        }

        if let Some(tool_calls) = tool_calls {
            let finish_reason = choice
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(|finish_reason| finish_reason.as_str())
                .unwrap_or("tool_calls");
            return build_tool_call_sse_response(tool_calls, finish_reason, model);
        }

        if let Some(function_call) = function_call {
            let finish_reason = choice
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(|finish_reason| finish_reason.as_str())
                .unwrap_or("function_call");
            return build_tool_call_sse_response(
                &serde_json::Value::Array(vec![function_call.clone()]),
                finish_reason,
                model,
            );
        }
    }

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(cached_json.to_owned()))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_streaming_request() {
        let body =
            br#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        assert!(is_streaming_request(body));
    }

    #[test]
    fn detects_non_streaming_request() {
        let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
        assert!(!is_streaming_request(body));

        let body2 = br#"{"model":"gpt-4o","stream":false,"messages":[]}"#;
        assert!(!is_streaming_request(body2));
    }

    #[tokio::test]
    async fn build_sse_contains_done_and_chunk_shape() {
        use http_body_util::BodyExt;

        let resp = build_sse_response("Hello world", "gpt-4o");
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"object\":\"chat.completion.chunk\""));
        assert!(text.contains("\"delta\":{\"role\":\"assistant\"}"));
        assert!(text.contains("\"delta\":{\"content\":\"Hello world\"}"));
        assert!(text.contains("data: [DONE]"));
    }

    #[test]
    fn cached_openai_json_converts_to_sse() {
        let cached = r#"{"choices":[{"message":{"role":"assistant","content":"cached answer"},"index":0,"finish_reason":"stop"}],"model":"gpt-4o"}"#;
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

    #[tokio::test]
    async fn cached_tool_call_json_converts_to_sse() {
        use http_body_util::BodyExt;

        let cached = r#"{
            "choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"index":0,"finish_reason":"tool_calls"}],
            "model":"gpt-4o"
        }"#;
        let resp = cached_to_sse_response(cached, "fallback-model");
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"tool_calls\":["));
        assert!(text.contains("\"finish_reason\":\"tool_calls\""));
        assert!(text.contains("data: [DONE]"));
    }
}

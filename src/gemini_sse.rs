//! Gemini GenerateContent SSE helpers.
//!
//! Gemini streaming uses `streamGenerateContent`; Isartor keeps canonical JSON
//! in cache and converts to SSE only at the boundary.

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde_json::{Value, json};

pub fn build_json_response(text: &str, model: &str) -> Value {
    json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": text}]
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "modelVersion": model
    })
}

pub fn build_sse_response(text: &str, model: &str) -> Response {
    let payload = build_json_response(text, model).to_string();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(Bytes::from(format!("data: {payload}\n\n"))))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

pub fn cached_to_sse_response(cached_json: &str, model_fallback: &str) -> Response {
    let payload = serde_json::from_str::<Value>(cached_json)
        .ok()
        .and_then(|value| {
            if value.get("candidates").is_some() {
                Some(value)
            } else {
                let text = value
                    .get("message")
                    .and_then(|message| message.as_str())
                    .map(ToOwned::to_owned)?;
                let model = value
                    .get("model")
                    .and_then(|model| model.as_str())
                    .unwrap_or(model_fallback);
                Some(build_json_response(&text, model))
            }
        })
        .unwrap_or_else(|| build_json_response(cached_json, model_fallback));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(Bytes::from(format!("data: {}\n\n", payload))))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    #[tokio::test]
    async fn cached_json_converts_to_gemini_sse() {
        let cached = build_json_response("hello", "gemini-2.0-flash").to_string();
        let response = cached_to_sse_response(&cached, "fallback");
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"candidates\""));
        assert!(text.contains("\"text\":\"hello\""));
    }
}

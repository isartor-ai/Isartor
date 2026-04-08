//! Cursor format adapter.
//!
//! Cursor sends standard OpenAI `/v1/chat/completions` requests with extra
//! headers (`X-Cursor-Checksum`, `X-Cursor-Client-Version`, `X-Ghost-Mode`).
//! The adapter is a thin wrapper around the OpenAI adapter — it strips the
//! cursor-specific headers on the way in and uses a separate cache namespace
//! so Cursor sessions don't pollute the generic OpenAI cache.

use axum::response::Response;

use super::ApiFormat;
use super::openai::{self, OpenAiFormat};
use super::types::{InternalRequest, InternalResponse};

pub struct CursorFormat;

impl ApiFormat for CursorFormat {
    fn name(&self) -> &'static str {
        "cursor"
    }

    fn cache_namespace(&self) -> &'static str {
        "cursor"
    }

    fn parse_request(&self, body: &[u8]) -> anyhow::Result<InternalRequest> {
        // Cursor uses identical body format to OpenAI; headers carry cursor-specific metadata.
        openai::parse_openai_body(body)
    }

    fn build_response(&self, resp: InternalResponse, streaming: bool) -> Response {
        // Cursor expects OpenAI-compatible responses.
        OpenAiFormat.build_response(resp, streaming)
    }
}

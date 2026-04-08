//! Kiro format adapter.
//!
//! Kiro is AWS's AI IDE. It uses an OpenAI-compatible wire format, so this
//! adapter starts as a thin wrapper around the OpenAI adapter with its own
//! cache namespace. Future versions may extend this as Kiro's protocol becomes
//! more fully documented.

use axum::response::Response;

use super::ApiFormat;
use super::openai::{self, OpenAiFormat};
use super::types::{InternalRequest, InternalResponse};

pub struct KiroFormat;

impl ApiFormat for KiroFormat {
    fn name(&self) -> &'static str {
        "kiro"
    }

    fn cache_namespace(&self) -> &'static str {
        "kiro"
    }

    fn parse_request(&self, body: &[u8]) -> anyhow::Result<InternalRequest> {
        openai::parse_openai_body(body)
    }

    fn build_response(&self, resp: InternalResponse, streaming: bool) -> Response {
        OpenAiFormat.build_response(resp, streaming)
    }
}

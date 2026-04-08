//! Format translation matrix for the Isartor gateway.
//!
//! All five client wire formats (OpenAI, Anthropic, Gemini, Cursor, Kiro)
//! implement the [`ApiFormat`] trait. Incoming requests are parsed into
//! [`InternalRequest`]; outgoing responses are built from [`InternalResponse`].
//! Provider-side translation lives in the [`translate`] submodule.
//!
//! ## Cache-key invariant
//! Each format returns a different [`ApiFormat::cache_namespace`] so that
//! OpenAI, Anthropic, Gemini, Cursor, and Kiro responses never share cache
//! entries.

pub mod anthropic;
pub mod cursor;
pub mod gemini;
pub mod kiro;
pub mod openai;
pub mod translate;
pub mod types;

use axum::http::HeaderMap;
use axum::response::Response;

pub use types::{
    InternalChunk, InternalContent, InternalMessage, InternalRequest, InternalResponse,
    InternalRole, InternalTool,
};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Wire-format adapter. Implement this for each client protocol Isartor serves.
pub trait ApiFormat: Send + Sync {
    /// Deserialise a raw request body into the canonical internal representation.
    fn parse_request(&self, body: &[u8]) -> anyhow::Result<InternalRequest>;

    /// Serialise an [`InternalResponse`] back to the client's expected wire
    /// format. `streaming` controls whether SSE or plain JSON is returned.
    fn build_response(&self, resp: InternalResponse, streaming: bool) -> Response;

    /// Return the cache namespace string, e.g. `"openai"`, `"anthropic"`, …
    fn cache_namespace(&self) -> &'static str;

    /// Human-readable format name for logs and metrics.
    fn name(&self) -> &'static str;
}

// ── Format detection ──────────────────────────────────────────────────────────

/// Detect which client format applies to an incoming request.
///
/// Decision order:
/// 1. Anthropic `/v1/messages`
/// 2. Gemini `/v1beta/models/…:generateContent|streamGenerateContent`
/// 3. Cursor (`X-Cursor-Checksum` header present)
/// 4. Kiro (`X-Kiro-*` header present)
/// 5. Default → OpenAI (covers `/v1/chat/completions` and native `/api/chat`)
pub fn detect_format(path: &str, headers: &HeaderMap) -> Box<dyn ApiFormat> {
    if path == "/v1/messages" {
        return Box::new(anthropic::AnthropicFormat);
    }
    if crate::core::prompt::is_gemini_path(path) {
        return Box::new(gemini::GeminiFormat);
    }
    if headers.contains_key("x-cursor-checksum")
        || headers.contains_key("x-cursor-client-version")
        || headers.contains_key("x-ghost-mode")
    {
        return Box::new(cursor::CursorFormat);
    }
    if headers.contains_key("x-kiro-version") || headers.contains_key("x-kiro-client-id") {
        return Box::new(kiro::KiroFormat);
    }
    Box::new(openai::OpenAiFormat)
}

/// Return only the cache namespace string without allocating a trait object.
///
/// Slightly more efficient than `detect_format(…).cache_namespace()` for the
/// common case where only the namespace is needed (cache middleware).
pub fn cache_namespace(path: &str, headers: &HeaderMap) -> &'static str {
    if path == "/v1/messages" {
        return "anthropic";
    }
    if crate::core::prompt::is_gemini_path(path) {
        return "gemini";
    }
    if headers.contains_key("x-cursor-checksum")
        || headers.contains_key("x-cursor-client-version")
        || headers.contains_key("x-ghost-mode")
    {
        return "cursor";
    }
    if headers.contains_key("x-kiro-version") || headers.contains_key("x-kiro-client-id") {
        return "kiro";
    }
    if path == "/v1/chat/completions" {
        "openai"
    } else {
        "native"
    }
}

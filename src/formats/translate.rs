//! Provider-side translation: convert [`InternalRequest`] into the wire format
//! expected by the target L3 provider.
//!
//! This is used by the passthrough path in `handler.rs`. Instead of forwarding
//! the raw client body (which may be in Gemini or Anthropic format) to an
//! OpenAI-compatible provider, we first parse into [`InternalRequest`] and then
//! call one of the `internal_to_*_body` functions below.

use serde_json::Value;

use crate::config::LlmProvider;

use super::anthropic::internal_to_anthropic_body;
use super::gemini::internal_to_gemini_body;
use super::openai::internal_to_openai_body;
use super::types::InternalRequest;

/// Target wire format for the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderWireFormat {
    /// OpenAI Chat Completions (`POST /v1/chat/completions`).
    OpenAi,
    /// Anthropic Messages API (`POST /v1/messages`).
    Anthropic,
    /// Google Gemini GenerateContent.
    Gemini,
}

/// Determine which wire format a given provider expects.
pub fn provider_wire_format(provider: &LlmProvider) -> ProviderWireFormat {
    match provider {
        LlmProvider::Anthropic => ProviderWireFormat::Anthropic,
        LlmProvider::Gemini => ProviderWireFormat::Gemini,
        _ => ProviderWireFormat::OpenAi,
    }
}

/// Translate an [`InternalRequest`] into the provider's expected JSON body.
///
/// Returns serialised JSON bytes ready to be forwarded as an HTTP request body.
pub fn translate_request(req: &InternalRequest, provider: &LlmProvider) -> anyhow::Result<Vec<u8>> {
    let body: Value = match provider_wire_format(provider) {
        ProviderWireFormat::OpenAi => internal_to_openai_body(req),
        ProviderWireFormat::Anthropic => internal_to_anthropic_body(req),
        ProviderWireFormat::Gemini => internal_to_gemini_body(req),
    };
    Ok(serde_json::to_vec(&body)?)
}

/// Translate an [`InternalRequest`] into an OpenAI-compatible JSON body.
///
/// Convenience wrapper used by `send_openai_passthrough_request`.
pub fn translate_request_to_openai(req: &InternalRequest) -> anyhow::Result<Vec<u8>> {
    translate_request(req, &LlmProvider::Openai)
}

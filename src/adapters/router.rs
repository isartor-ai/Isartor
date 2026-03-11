#![allow(dead_code)]
//! # Router Adapters — Concrete implementations of `SlmRouter`
//!
//! | Adapter                | Inference Backend            | Use Case                    |
//! |------------------------|------------------------------|-----------------------------|
//! | `EmbeddedCandleRouter` | Candle (in-process GGUF)     | Minimalist / edge / single  |
//! | `RemoteVllmRouter`     | vLLM / TGI (HTTP)            | Enterprise / GPU cluster    |

use async_trait::async_trait;
use tracing::Instrument;

use crate::core::ports::SlmRouter;

// ═══════════════════════════════════════════════════════════════════════
// Adapter: EmbeddedCandleRouter — in-process quantised model via Candle
// ═══════════════════════════════════════════════════════════════════════

/// In-process intent classifier powered by the Candle ML framework.
///
/// Loads a quantised GGUF model (e.g. Gemma-2-2B-IT Q4_K_M) directly
/// into the process and performs inference on CPU without any network hop.
///
/// **Note:** This skeleton mocks the model state.  In production, this
/// struct would hold `Arc<Mutex<candle_transformers::...::ModelWeights>>`
/// and a `tokenizers::Tokenizer`.
pub struct EmbeddedCandleRouter {
    /// HuggingFace repository ID (e.g. `mradermacher/gemma-2-2b-it-GGUF`).
    _repo_id: String,
    /// GGUF model filename (e.g. `gemma-2-2b-it.Q4_K_M.gguf`).
    _gguf_filename: String,
    // In a full implementation:
    // model: Arc<tokio::sync::Mutex<ModelWeights>>,
    // tokenizer: Arc<Tokenizer>,
}

impl EmbeddedCandleRouter {
    /// Create a new embedded Candle router (skeleton).
    ///
    /// In production this would download and load the GGUF model into
    /// memory.  For now it stores the configuration.
    pub fn new(repo_id: impl Into<String>, gguf_filename: impl Into<String>) -> Self {
        let _repo_id = repo_id.into();
        let _gguf_filename = gguf_filename.into();
        log::info!(
            "EmbeddedCandleRouter adapter created (skeleton) repo={} file={}",
            _repo_id,
            _gguf_filename
        );
        Self {
            _repo_id,
            _gguf_filename,
        }
    }
}

#[async_trait]
impl SlmRouter for EmbeddedCandleRouter {
    async fn classify_intent(&self, prompt: &str) -> anyhow::Result<String> {
        let span = tracing::info_span!(
            "l2_classify_intent",
            router.backend = "embedded_candle",
            router.decision = tracing::field::Empty,
            prompt_len = prompt.len(),
        );
        let _guard = span.enter();
        // Skeleton: tokenise → forward pass → parse LABEL from output.
        // Gated behind `embedded-inference` feature flag; not yet wired.
        tracing::debug!("EmbeddedCandleRouter: classify_intent (skeleton)");
        // Default classification: fall through to the external LLM.
        let decision = "COMPLEX";
        span.record("router.decision", decision);
        tracing::info!(
            router.decision = decision,
            "L2 intent classified (embedded candle)"
        );
        Ok(decision.to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Adapter: RemoteVllmRouter — remote model serving via HTTP (vLLM / TGI)
// ═══════════════════════════════════════════════════════════════════════

/// Intent classifier that delegates to a remote vLLM (or compatible)
/// inference server over HTTP.
///
/// Sends an OpenAI-compatible `/v1/chat/completions` request to the
/// configured endpoint and parses the LABEL from the response.
///
/// This is the enterprise adapter for GPU-cluster deployments where
/// models are served externally on dedicated GPU nodes.
pub struct RemoteVllmRouter {
    /// HTTP client for outbound requests.
    client: reqwest::Client,
    /// Base URL of the vLLM server (e.g. `http://vllm.gpu-pool:8000`).
    base_url: String,
    /// Model name sent in the `"model"` field of the OpenAI-compatible
    /// request payload.
    model_name: String,
}

impl RemoteVllmRouter {
    /// Create a new remote vLLM router adapter.
    ///
    /// # Arguments
    /// * `client`     — Shared `reqwest::Client` (reuse connection pools).
    /// * `base_url`   — vLLM / TGI base URL.
    /// * `model_name` — Model identifier for the API request.
    pub fn new(
        client: reqwest::Client,
        base_url: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        let base_url = base_url.into();
        let model_name = model_name.into();
        log::info!(
            "RemoteVllmRouter adapter created (skeleton) url={} model={}",
            base_url,
            model_name
        );
        Self {
            client,
            base_url,
            model_name,
        }
    }
}

#[async_trait]
impl SlmRouter for RemoteVllmRouter {
    async fn classify_intent(&self, prompt: &str) -> anyhow::Result<String> {
        let span = tracing::info_span!(
            "l2_classify_intent",
            router.backend = "remote_vllm",
            router.decision = tracing::field::Empty,
            router.model = %self.model_name,
            router.url = %self.base_url,
            prompt_len = prompt.len(),
        );
        async {
            // System prompt for intent classification
            let system_prompt = "You are an intent classifier. Given a user prompt, respond with one word: either 'SIMPLE' or 'COMPLEX'. Only output the label.";

            let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
            let req_body = serde_json::json!({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1,
                "temperature": 0.0
            });

            let resp = self.client.post(&url)
                .json(&req_body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("RemoteVllmRouter HTTP error: {}: {}", status, text);
            }

            let resp_json: serde_json::Value = resp.json().await?;
            // Try to extract the label from the first choice
            let label = resp_json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_uppercase();
            let decision = if label == "SIMPLE" || label == "COMPLEX" {
                label
            } else {
                // Fallback: treat as COMPLEX if label is not recognized
                "COMPLEX".to_string()
            };
            tracing::Span::current().record("router.decision", decision.as_str());
            tracing::info!(router.decision = %decision, "L2 intent classified (remote vLLM)");
            Ok(decision)
        }
        .instrument(span)
        .await
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embedded_candle_skeleton_returns_complex() {
        let router = EmbeddedCandleRouter::new(
            "mradermacher/gemma-2-2b-it-GGUF",
            "gemma-2-2b-it.Q4_K_M.gguf",
        );
        let label = router.classify_intent("Hello world").await.unwrap();
        assert_eq!(label, "COMPLEX");
    }

    #[tokio::test]
    async fn remote_vllm_skeleton_returns_complex() {
        // Use wiremock to mock the vLLM endpoint
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let mock_server = MockServer::start().await;
        // Mock the /v1/chat/completions endpoint to return a COMPLEX label
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": "COMPLEX" }
                }]
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let router = RemoteVllmRouter::new(client, mock_server.uri(), "gemma-2-2b-it");
        let label = router
            .classify_intent("Explain quantum computing")
            .await
            .unwrap();
        assert_eq!(label, "COMPLEX");
    }
}

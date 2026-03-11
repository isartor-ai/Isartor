#![allow(dead_code)]
//! # Router Adapters — Concrete implementations of `SlmRouter`
//!
//! | Adapter                | Inference Backend            | Use Case                    |
//! |------------------------|------------------------------|-----------------------------|
//! | `EmbeddedCandleRouter` | Candle (in-process GGUF)     | Minimalist / edge / single  |
//! | `RemoteVllmRouter`     | vLLM / TGI (HTTP)            | Enterprise / GPU cluster    |

use async_trait::async_trait;

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
        // TODO: Tokenise → forward pass → parse LABEL from output.
        log::debug!(
            "EmbeddedCandleRouter::classify_intent (skeleton) prompt_len={}",
            prompt.len()
        );
        // Default classification: fall through to the external LLM.
        Ok("COMPLEX".to_string())
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
        // TODO: POST to {base_url}/v1/chat/completions with the
        //       classification system prompt and parse the LABEL.
        log::debug!(
            "RemoteVllmRouter::classify_intent (skeleton) prompt_len={} url={} model={}",
            prompt.len(),
            self.base_url,
            self.model_name
        );
        // Skeleton: return COMPLEX so the pipeline always falls through
        // to the external LLM until the real HTTP call is wired.
        let _ = &self.client; // suppress unused-field warning
        Ok("COMPLEX".to_string())
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
        let client = reqwest::Client::new();
        let router = RemoteVllmRouter::new(client, "http://localhost:8000", "gemma-2-2b-it");
        let label = router
            .classify_intent("Explain quantum computing")
            .await
            .unwrap();
        assert_eq!(label, "COMPLEX");
    }
}

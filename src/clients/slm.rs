#![allow(dead_code)]
#![allow(clippy::trim_split_whitespace)]
// =============================================================================
// SlmClient — HTTP client for the llama.cpp sidecar's OpenAI-compatible API.
//
// The llama.cpp server exposes OpenAI-compatible endpoints:
//   POST /v1/chat/completions  — text generation
//   POST /v1/embeddings        — embedding generation
//   GET  /health               — health check
//
// This client wraps the generation endpoint. The embedding endpoint
// is consumed directly by the `LlamaCppEmbedder` pipeline component.
// =============================================================================

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::Layer2Settings;

// ── OpenAI-compatible request / response types ───────────────────────

/// A single message in the OpenAI Chat Completions format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    /// Sampling temperature. Lower = more deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// A single choice from the completions response.
#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    #[allow(dead_code)]
    pub index: u32,
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

/// Response body from `POST /v1/chat/completions`.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
    #[allow(dead_code)]
    pub model: Option<String>,
}

// ── SlmClient ────────────────────────────────────────────────────────

/// Dedicated HTTP client for the Layer 2 llama.cpp sidecar.
///
/// Holds a pre-configured `reqwest::Client` with the appropriate timeout
/// and the base URL of the sidecar server.
///
/// # Example
///
/// ```ignore
/// let client = SlmClient::new(&settings.layer2);
/// let category = client.classify_intent("Write a poem about Rust").await?;
/// assert!(category.contains("CREATIVE_WRITING"));
/// ```
#[derive(Clone)]
pub struct SlmClient {
    http_client: reqwest::Client,
    base_url: String,
    model_name: String,
}

/// Classification system prompt — instructs the SLM to emit exactly
/// one category label with no surrounding explanation.
const CLASSIFY_SYSTEM_PROMPT: &str = "\
You are a classification engine. Classify the user prompt into exactly one of \
these categories: [SIMPLE_LOOKUP, CODING_TASK, CREATIVE_WRITING, COMPLEX_REASONING]. \
Return only the category name and nothing else.";

const SIMPLE_COMPLEX_SYSTEM_PROMPT: &str = "\
You are a request classifier. Decide whether the following user prompt \
is SIMPLE (can be answered with a short factual response, a greeting, or \
basic knowledge) or COMPLEX (requires deep reasoning, code generation, \
creative writing, or multi-step analysis).\n\n\
Reply with EXACTLY one word: SIMPLE or COMPLEX.";

impl SlmClient {
    /// Create a new `SlmClient` from the application's `Layer2Settings`.
    ///
    /// Initialises a dedicated `reqwest::Client` with the configured
    /// timeout so it does not share connection pools with other services.
    pub fn new(settings: &Layer2Settings) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(settings.timeout_seconds))
            .build()
            .expect("failed to build SlmClient reqwest::Client");

        Self {
            http_client,
            base_url: settings.sidecar_url.trim_end_matches('/').to_string(),
            model_name: settings.model_name.clone(),
        }
    }

    /// Classify the user's intent via the local SLM.
    ///
    /// Sends the prompt to the llama.cpp sidecar's OpenAI-compatible
    /// `/v1/chat/completions` endpoint with a constraining system
    /// prompt. Returns the raw classification label string.
    pub async fn classify_intent(&self, user_prompt: &str) -> anyhow::Result<String> {
        let request_body = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: CLASSIFY_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                },
            ],
            temperature: Some(0.0), // Deterministic classification.
            max_tokens: Some(20),   // Category name only.
            stream: Some(false),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let resp = self
            .http_client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("SlmClient: sidecar request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("SlmClient: sidecar returned {status}: {body}");
        }

        let completion: ChatCompletionResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("SlmClient: failed to parse response: {e}"))?;

        let content = completion
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .unwrap_or_default();

        tracing::debug!(
            classification = %content,
            model = %self.model_name,
            "SlmClient: intent classified"
        );

        Ok(content)
    }

    /// Send an arbitrary chat completion request to the sidecar.
    ///
    /// This is the general-purpose method used by pipeline components
    /// (intent classifier, local executor, reranker) that need full
    /// control over the message list and sampling parameters.
    pub async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<String> {
        let request_body = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages,
            temperature,
            max_tokens,
            stream: Some(false),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let resp = self
            .http_client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("SlmClient: chat request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("SlmClient: chat returned {status}: {body}");
        }

        let completion: ChatCompletionResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("SlmClient: failed to parse chat response: {e}"))?;

        let content = completion
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        Ok(content)
    }

    /// Returns true when the sidecar classifies the prompt as SIMPLE.
    pub async fn classify_simple_or_complex(&self, user_prompt: &str) -> anyhow::Result<bool> {
        let request_body = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: SIMPLE_COMPLEX_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                },
            ],
            temperature: Some(0.0),
            max_tokens: Some(10),
            stream: Some(false),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .http_client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("SlmClient: simple/complex request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("SlmClient: simple/complex returned {status}: {body}");
        }

        let completion: ChatCompletionResponse = resp.json().await.map_err(|e| {
            anyhow::anyhow!("SlmClient: failed to parse simple/complex response: {e}")
        })?;

        let content = completion
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default()
            .trim()
            .to_uppercase();

        Ok(content.contains("SIMPLE"))
    }

    /// Classifier-mode-aware deflection check.
    ///
    /// In **tiered** mode, sends the tiered system prompt and returns
    /// `true` when the SLM replies TEMPLATE or SNIPPET.
    /// In **binary** mode, delegates to `classify_simple_or_complex`.
    pub async fn classify_deflectable(
        &self,
        user_prompt: &str,
        mode: &crate::config::ClassifierMode,
    ) -> anyhow::Result<bool> {
        if *mode == crate::config::ClassifierMode::Binary {
            return self.classify_simple_or_complex(user_prompt).await;
        }

        let system_prompt = crate::middleware::slm_triage::CLASSIFY_TIERED_SYSTEM_PROMPT;
        let request_body = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                },
            ],
            temperature: Some(0.0),
            max_tokens: Some(10),
            stream: Some(false),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .http_client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("SlmClient: tiered classify request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("SlmClient: tiered classify returned {status}: {body}");
        }

        let completion: ChatCompletionResponse = resp.json().await.map_err(|e| {
            anyhow::anyhow!("SlmClient: failed to parse tiered classify response: {e}")
        })?;

        let content = completion
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default()
            .trim()
            .to_uppercase();

        Ok(content.contains("TEMPLATE") || content.contains("SNIPPET"))
    }

    pub async fn answer_prompt(&self, user_prompt: &str) -> anyhow::Result<String> {
        self.chat_completion(
            vec![ChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }],
            None,
            None,
        )
        .await
    }

    /// Returns the base URL of the sidecar.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns a reference to the underlying HTTP client (for reuse by
    /// pipeline components that need the same timeout/pool settings).
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_settings(url: &str) -> Layer2Settings {
        Layer2Settings {
            sidecar_url: url.to_string(),
            model_name: "test-model".to_string(),
            timeout_seconds: 5,
            classifier_mode: crate::config::ClassifierMode::Tiered,
            max_answer_tokens: 2048,
        }
    }

    fn mock_completion_response(content: &str) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "index": 0,
                "finish_reason": "stop"
            }],
            "model": "test-model"
        })
    }

    // ── Construction tests ───────────────────────────────────────

    #[test]
    fn slm_client_construction() {
        let settings = test_settings("http://localhost:8081/");
        let client = SlmClient::new(&settings);
        assert_eq!(client.base_url(), "http://localhost:8081");
        assert_eq!(client.model_name(), "test-model");
    }

    #[test]
    fn slm_client_trims_trailing_slash() {
        let settings = test_settings("http://host:1234///");
        let client = SlmClient::new(&settings);
        assert_eq!(client.base_url(), "http://host:1234");
    }

    // ── classify_intent tests ────────────────────────────────────

    #[tokio::test]
    async fn classify_intent_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mock_completion_response("SIMPLE_LOOKUP")),
            )
            .mount(&server)
            .await;

        let client = SlmClient::new(&test_settings(&server.uri()));
        let result = client.classify_intent("What is 2+2?").await.unwrap();
        assert_eq!(result, "SIMPLE_LOOKUP");
    }

    #[tokio::test]
    async fn classify_intent_server_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal error"))
            .mount(&server)
            .await;

        let client = SlmClient::new(&test_settings(&server.uri()));
        let result = client.classify_intent("hello").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("500"));
    }

    #[tokio::test]
    async fn classify_intent_empty_choices() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"choices": [], "model": "test"})),
            )
            .mount(&server)
            .await;

        let client = SlmClient::new(&test_settings(&server.uri()));
        let result = client.classify_intent("hello").await.unwrap();
        assert_eq!(result, "");
    }

    // ── chat_completion tests ────────────────────────────────────

    #[tokio::test]
    async fn chat_completion_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mock_completion_response("Hello there!")),
            )
            .mount(&server)
            .await;

        let client = SlmClient::new(&test_settings(&server.uri()));
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        }];
        let result = client
            .chat_completion(messages, Some(0.7), Some(100))
            .await
            .unwrap();
        assert_eq!(result, "Hello there!");
    }

    #[tokio::test]
    async fn chat_completion_server_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let client = SlmClient::new(&test_settings(&server.uri()));
        let result = client.chat_completion(vec![], None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn chat_completion_malformed_response() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not valid json"))
            .mount(&server)
            .await;

        let client = SlmClient::new(&test_settings(&server.uri()));
        let result = client.chat_completion(vec![], None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    // ── Serialization tests ──────────────────────────────────────

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: "system".into(),
            content: "test".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "system");
        assert_eq!(back.content, "test");
    }

    #[test]
    fn chat_completion_request_skips_none_fields() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("temperature"));
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("stream"));
    }

    #[test]
    fn chat_completion_response_deserialize() {
        let json = r#"{
            "choices": [{
                "message": {"role": "assistant", "content": "Hi"},
                "index": 0,
                "finish_reason": "stop"
            }],
            "model": "phi-3"
        }"#;
        let resp: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content, "Hi");
        assert_eq!(resp.model, Some("phi-3".into()));
    }
}

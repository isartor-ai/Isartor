#![allow(dead_code)]
#![allow(clippy::trim_split_whitespace)]
use serde::{Deserialize, Serialize};

// ── Observability: Final-layer annotation ────────────────────────────

/// Which firewall layer ultimately resolved the request.
/// Inserted into `http::Extensions` by each middleware so the root
/// monitoring span can record it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalLayer {
    /// Layer 0 — request rejected by auth / rate-limiter.
    AuthBlocked,
    /// Layer 1a — exact (SHA-256) cache hit.
    ExactCache,
    /// Layer 1b — semantic (embedding + cosine) cache hit.
    SemanticCache,
    /// Layer 2 — local SLM answered the request.
    Slm,
    /// Layer 3 — external cloud LLM fallback.
    Cloud,
}

impl FinalLayer {
    /// Stable string label used in OTel metrics and span attributes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AuthBlocked => "L0_AuthBlocked",
            Self::ExactCache => "L1a_ExactCache",
            Self::SemanticCache => "L1b_SemanticCache",
            Self::Slm => "L2_SLM",
            Self::Cloud => "L3_Cloud",
        }
    }

    /// Stable short label used in response headers and operator views.
    pub fn as_header_value(&self) -> &'static str {
        match self {
            Self::AuthBlocked => "l0",
            Self::ExactCache => "l1a",
            Self::SemanticCache => "l1b",
            Self::Slm => "l2",
            Self::Cloud => "l3",
        }
    }

    pub fn is_deflected(&self) -> bool {
        matches!(
            self,
            Self::AuthBlocked | Self::ExactCache | Self::SemanticCache | Self::Slm
        )
    }
}

// ── Client ↔ Firewall ─────────────────────────────────────────────────

/// Incoming chat request from a client.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
pub struct ChatRequest {
    pub prompt: String,
}

/// Unified response envelope returned by any layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub layer: u8,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// A recent routing decision for intercepted proxy traffic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyRouteDecision {
    pub request_id: String,
    pub timestamp: String,
    pub client: String,
    pub hostname: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_hash: Option<String>,
    pub final_layer: String,
    pub resolved_by: String,
    pub deflected: bool,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyRecentResponse {
    pub entries: Vec<ProxyRouteDecision>,
}

// ── Legacy Ollama — Generation (v1 middleware compat) ─────────────────

/// Request body for Ollama `/api/generate`.
#[derive(Debug, Serialize)]
pub struct OllamaRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
}

/// Response from Ollama `/api/generate` (non-streaming).
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    pub response: String,
}

// ── Legacy Ollama — Embeddings (v1 middleware compat) ─────────────────

/// Request body for Ollama `/api/embed`.
#[derive(Debug, Serialize)]
pub struct OllamaEmbedRequest {
    pub model: String,
    pub input: String,
}

/// Response from Ollama `/api/embed`.
#[derive(Debug, Deserialize)]
pub struct OllamaEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

// ═════════════════════════════════════════════════════════════════════
// OpenAI-Compatible Types — used by llama.cpp sidecar (v2 pipeline)
// ═════════════════════════════════════════════════════════════════════

/// A single message in the OpenAI Chat Completions format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: String,
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Serialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// A single choice from the chat completions response.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAiChatChoice {
    pub message: OpenAiMessage,
    #[allow(dead_code)]
    pub index: u32,
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

/// Response body from `POST /v1/chat/completions`.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAiChatResponse {
    pub choices: Vec<OpenAiChatChoice>,
    #[allow(dead_code)]
    pub model: Option<String>,
}

/// Request body for `POST /v1/embeddings`.
#[derive(Debug, Serialize)]
pub struct OpenAiEmbeddingRequest {
    pub model: String,
    pub input: String,
}

/// A single embedding object in the response.
#[derive(Debug, Deserialize)]
pub struct OpenAiEmbeddingData {
    pub embedding: Vec<f32>,
    #[allow(dead_code)]
    pub index: u32,
}

/// Response body from `POST /v1/embeddings`.
#[derive(Debug, Deserialize)]
pub struct OpenAiEmbeddingResponse {
    pub data: Vec<OpenAiEmbeddingData>,
    #[allow(dead_code)]
    pub model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChatRequest ──────────────────────────────────────────────

    #[test]
    fn chat_request_deserialize() {
        let json = r#"{"prompt":"Hello world"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "Hello world");
    }

    #[test]
    fn chat_request_serialize_roundtrip() {
        let req = ChatRequest {
            prompt: "test".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt, "test");
    }

    // ── ChatResponse ─────────────────────────────────────────────

    #[test]
    fn chat_response_serialize_with_model() {
        let resp = ChatResponse {
            layer: 3,
            message: "Hello".into(),
            model: Some("gpt-4o".into()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"layer\":3"));
        assert!(json.contains("\"model\":\"gpt-4o\""));
    }

    #[test]
    fn chat_response_serialize_without_model_skips_field() {
        let resp = ChatResponse {
            layer: 2,
            message: "Simple answer".into(),
            model: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("\"model\""));
    }

    // ── OllamaRequest ───────────────────────────────────────────

    #[test]
    fn ollama_request_serialize() {
        let req = OllamaRequest {
            model: "llama3".into(),
            prompt: "hello".into(),
            stream: false,
            system: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"stream\":false"));
        assert!(!json.contains("\"system\""));
    }

    #[test]
    fn ollama_request_serialize_with_system() {
        let req = OllamaRequest {
            model: "llama3".into(),
            prompt: "hello".into(),
            stream: false,
            system: Some("You are helpful.".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"system\":\"You are helpful.\""));
    }

    // ── OllamaResponse ──────────────────────────────────────────

    #[test]
    fn ollama_response_deserialize() {
        let json = r#"{"response":"Hi there!"}"#;
        let resp: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response, "Hi there!");
    }

    // ── OllamaEmbedRequest / Response ────────────────────────────

    #[test]
    fn ollama_embed_request_serialize() {
        let req = OllamaEmbedRequest {
            model: "all-minilm".into(),
            input: "test".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"all-minilm\""));
    }

    #[test]
    fn ollama_embed_response_deserialize() {
        let json = r#"{"embeddings":[[0.1, 0.2, 0.3]]}"#;
        let resp: OllamaEmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.embeddings.len(), 1);
        assert_eq!(resp.embeddings[0].len(), 3);
    }

    // ── OpenAI Chat Types ────────────────────────────────────────

    #[test]
    fn openai_message_roundtrip() {
        let msg = OpenAiMessage {
            role: "user".into(),
            content: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: OpenAiMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content, "hello");
    }

    #[test]
    fn openai_chat_request_serialize_minimal() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".into(),
            messages: vec![OpenAiMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        // skip_serializing_if = None should omit optional fields
        assert!(!json.contains("\"temperature\""));
        assert!(!json.contains("\"max_tokens\""));
        assert!(!json.contains("\"stream\""));
    }

    #[test]
    fn openai_chat_request_serialize_full() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: Some(false),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"max_tokens\":100"));
        assert!(json.contains("\"stream\":false"));
    }

    #[test]
    fn openai_chat_response_deserialize() {
        let json = r#"{
            "choices": [{
                "message": {"role": "assistant", "content": "Hi!"},
                "index": 0,
                "finish_reason": "stop"
            }],
            "model": "gpt-4o-mini"
        }"#;
        let resp: OpenAiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content, "Hi!");
        assert_eq!(resp.model, Some("gpt-4o-mini".into()));
    }

    #[test]
    fn openai_chat_response_deserialize_no_model() {
        let json = r#"{"choices": [{"message": {"role":"assistant","content":"ok"}, "index":0, "finish_reason": null}]}"#;
        let resp: OpenAiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.content, "ok");
        assert!(resp.model.is_none());
    }

    // ── OpenAI Embedding Types ───────────────────────────────────

    #[test]
    fn openai_embedding_request_serialize() {
        let req = OpenAiEmbeddingRequest {
            model: "all-minilm".into(),
            input: "test text".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"all-minilm\""));
        assert!(json.contains("\"input\":\"test text\""));
    }

    #[test]
    fn openai_embedding_response_deserialize() {
        let json = r#"{
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "all-minilm"
        }"#;
        let resp: OpenAiEmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].embedding, vec![0.1f32, 0.2, 0.3]);
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.model, Some("all-minilm".into()));
    }

    #[test]
    fn openai_embedding_response_no_model() {
        let json = r#"{"data": [{"embedding": [1.0], "index": 0}]}"#;
        let resp: OpenAiEmbeddingResponse = serde_json::from_str(json).unwrap();
        assert!(resp.model.is_none());
    }
}

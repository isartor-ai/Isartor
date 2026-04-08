//! Canonical internal types for the format translation matrix.
//!
//! All five client formats (OpenAI, Anthropic, Gemini, Cursor, Kiro) parse into
//! these types. Provider-side translation builds from these types into the
//! provider's expected wire format.

use serde::{Deserialize, Serialize};

// ── Message roles ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InternalRole {
    System,
    User,
    Assistant,
    Tool,
}

impl InternalRole {
    pub fn as_openai_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    pub fn as_gemini_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::System => "user",
            Self::Assistant => "model",
            Self::Tool => "user",
        }
    }
}

// ── Content parts ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InternalContent {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        detail: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

impl InternalContent {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }
}

// ── Tool definition ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

// ── Single message ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalMessage {
    pub role: InternalRole,
    pub content: Vec<InternalContent>,
}

impl InternalMessage {
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("")
    }
}

// ── Full request ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalRequest {
    pub model: String,
    /// System/developer instruction, extracted from whichever position the
    /// client format uses (OpenAI inline, Anthropic top-level, Gemini
    /// `systemInstruction`).
    pub system: Option<String>,
    pub messages: Vec<InternalMessage>,
    pub tools: Vec<InternalTool>,
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
}

impl InternalRequest {
    /// Build a stable prompt string for cache-key purposes.
    pub fn to_prompt(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(sys) = self.system.as_deref().filter(|s| !s.is_empty()) {
            parts.push(format!("[system] {sys}"));
        }
        for msg in &self.messages {
            let text = msg.text_content();
            if !text.is_empty() {
                let role = msg.role.as_openai_str();
                parts.push(format!("[{role}] {text}"));
            }
        }
        parts.join("\n")
    }

    /// Returns the last user message text, used for semantic (L1b) cache key.
    pub fn last_user_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == InternalRole::User)
            .map(|m| m.text_content())
            .filter(|s| !s.is_empty())
    }
}

// ── Response ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalResponse {
    pub model: String,
    pub content: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub stop_reason: Option<String>,
}

// ── SSE chunk ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InternalChunk {
    pub model: String,
    pub delta: String,
    pub finish_reason: Option<String>,
}

//! # Firewall Error Types
//!
//! Provides a structured error hierarchy that distinguishes **retryable**
//! from **fatal** errors. Every layer in the Deflection Stack maps its failures
//! into [`GatewayError`] so the handler and middleware can make
//! retry/fallback decisions consistently.

use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
// Error Classification
// ═══════════════════════════════════════════════════════════════════════

/// Whether an error is worth retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Transient failure — network timeout, 429 rate limit, 502/503/504
    /// from an upstream. These should be retried with back-off.
    Retryable,

    /// Permanent failure — invalid API key (401/403), malformed request
    /// (400), missing configuration. Retrying will not help.
    Fatal,
}

// ═══════════════════════════════════════════════════════════════════════
// Gateway Error
// ═══════════════════════════════════════════════════════════════════════

/// Unified error type for the Isartor gateway pipeline.
#[derive(Debug)]
pub enum GatewayError {
    /// Layer 1 cache subsystem error (exact or semantic).
    Cache {
        layer: &'static str,
        message: String,
        class: ErrorClass,
    },

    /// Layer 1b embedding generation error.
    Embedding { message: String, class: ErrorClass },

    /// Layer 2 SLM inference / classification error.
    Inference { message: String, class: ErrorClass },

    /// Layer 3 cloud LLM call error.
    CloudLlm {
        provider: String,
        message: String,
        class: ErrorClass,
    },

    /// Request validation error (bad body, missing fields, etc.).
    Validation { message: String },

    /// Internal misconfiguration (missing state, missing extensions).
    Configuration { message: String },
}

impl GatewayError {
    /// Returns the [`ErrorClass`] for this error.
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Cache { class, .. } => *class,
            Self::Embedding { class, .. } => *class,
            Self::Inference { class, .. } => *class,
            Self::CloudLlm { class, .. } => *class,
            // Validation and config errors are always fatal.
            Self::Validation { .. } => ErrorClass::Fatal,
            Self::Configuration { .. } => ErrorClass::Fatal,
        }
    }

    /// `true` if this error is worth retrying.
    pub fn is_retryable(&self) -> bool {
        self.class() == ErrorClass::Retryable
    }

    /// The gateway layer that produced the error, as a short label.
    pub fn layer_label(&self) -> &str {
        match self {
            Self::Cache { layer, .. } => layer,
            Self::Embedding { .. } => "L1b_Embedding",
            Self::Inference { .. } => "L2_SLM",
            Self::CloudLlm { .. } => "L3_Cloud",
            Self::Validation { .. } => "Validation",
            Self::Configuration { .. } => "Configuration",
        }
    }

    // ── Constructors ─────────────────────────────────────────────────

    /// Classify a cloud LLM error by inspecting the error message for
    /// common patterns (status codes, keywords).
    pub fn from_llm_error(provider: &str, err: &anyhow::Error) -> Self {
        let msg = err.to_string();
        let class = classify_error_message(&msg);
        Self::CloudLlm {
            provider: provider.to_string(),
            message: msg,
            class,
        }
    }

    /// Classify an HTTP-based inference error.
    pub fn from_inference_error(err: &anyhow::Error) -> Self {
        let msg = err.to_string();
        let class = classify_error_message(&msg);
        Self::Inference {
            message: msg,
            class,
        }
    }

    /// Cache error — always retryable (downstream layers can compensate).
    pub fn cache_error(layer: &'static str, msg: impl Into<String>) -> Self {
        Self::Cache {
            layer,
            message: msg.into(),
            class: ErrorClass::Retryable,
        }
    }

    /// Embedding error — retryable (skip semantic cache on failure).
    pub fn embedding_error(msg: impl Into<String>) -> Self {
        Self::Embedding {
            message: msg.into(),
            class: ErrorClass::Retryable,
        }
    }
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache { layer, message, .. } => {
                write!(f, "[{layer}] cache error: {message}")
            }
            Self::Embedding { message, .. } => {
                write!(f, "[embedding] {message}")
            }
            Self::Inference { message, .. } => {
                write!(f, "[inference] {message}")
            }
            Self::CloudLlm {
                provider, message, ..
            } => {
                write!(f, "[{provider}] {message}")
            }
            Self::Validation { message } => {
                write!(f, "[validation] {message}")
            }
            Self::Configuration { message } => {
                write!(f, "[config] {message}")
            }
        }
    }
}

impl std::error::Error for GatewayError {}

// ═══════════════════════════════════════════════════════════════════════
// Heuristic error classification
// ═══════════════════════════════════════════════════════════════════════

/// Inspect an error message string and decide whether the error is
/// retryable. This is a best-effort heuristic; provider-specific errors
/// are matched by common keywords and HTTP status codes.
fn classify_error_message(msg: &str) -> ErrorClass {
    let lower = msg.to_lowercase();

    // ── Fatal patterns ───────────────────────────────────────────
    let fatal_patterns = [
        "401",
        "403",
        "400",
        "invalid api key",
        "invalid_api_key",
        "authentication",
        "unauthorized",
        "forbidden",
        "invalid request",
        "invalid_request",
        "model not found",
        "model_not_found",
        "deployment not found",
    ];

    for pat in &fatal_patterns {
        if lower.contains(pat) {
            return ErrorClass::Fatal;
        }
    }

    // ── Retryable patterns ───────────────────────────────────────
    let retryable_patterns = [
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "429",
        "rate limit",
        "rate_limit",
        "502",
        "503",
        "504",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "temporarily unavailable",
        "network",
        "broken pipe",
        "eof",
        "dns",
        "resolve",
    ];

    for pat in &retryable_patterns {
        if lower.contains(pat) {
            return ErrorClass::Retryable;
        }
    }

    // Default: treat unknown errors as retryable so we at least try once
    // more before giving up.
    ErrorClass::Retryable
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_401_as_fatal() {
        let err = anyhow::anyhow!("HTTP 401 Unauthorized");
        let gw = GatewayError::from_llm_error("openai", &err);
        assert_eq!(gw.class(), ErrorClass::Fatal);
        assert!(!gw.is_retryable());
    }

    #[test]
    fn classify_invalid_api_key_as_fatal() {
        let err = anyhow::anyhow!("Error: invalid api key provided");
        let gw = GatewayError::from_llm_error("azure", &err);
        assert_eq!(gw.class(), ErrorClass::Fatal);
    }

    #[test]
    fn classify_timeout_as_retryable() {
        let err = anyhow::anyhow!("request timed out after 30s");
        let gw = GatewayError::from_llm_error("openai", &err);
        assert_eq!(gw.class(), ErrorClass::Retryable);
        assert!(gw.is_retryable());
    }

    #[test]
    fn classify_429_as_retryable() {
        let err = anyhow::anyhow!("HTTP 429 Too Many Requests");
        let gw = GatewayError::from_llm_error("anthropic", &err);
        assert_eq!(gw.class(), ErrorClass::Retryable);
    }

    #[test]
    fn classify_connection_refused_as_retryable() {
        let err = anyhow::anyhow!("connection refused");
        let gw = GatewayError::from_llm_error("xai", &err);
        assert_eq!(gw.class(), ErrorClass::Retryable);
    }

    #[test]
    fn classify_502_as_retryable() {
        let err = anyhow::anyhow!("Bad Gateway (502)");
        let gw = GatewayError::from_llm_error("openai", &err);
        assert_eq!(gw.class(), ErrorClass::Retryable);
    }

    #[test]
    fn classify_unknown_as_retryable() {
        let err = anyhow::anyhow!("something weird happened");
        let gw = GatewayError::from_llm_error("openai", &err);
        assert_eq!(gw.class(), ErrorClass::Retryable);
    }

    #[test]
    fn validation_error_is_fatal() {
        let gw = GatewayError::Validation {
            message: "missing prompt field".into(),
        };
        assert_eq!(gw.class(), ErrorClass::Fatal);
    }

    #[test]
    fn configuration_error_is_fatal() {
        let gw = GatewayError::Configuration {
            message: "missing state".into(),
        };
        assert_eq!(gw.class(), ErrorClass::Fatal);
    }

    #[test]
    fn cache_error_is_retryable() {
        let gw = GatewayError::cache_error("L1a_ExactCache", "Redis timeout");
        assert!(gw.is_retryable());
        assert_eq!(gw.layer_label(), "L1a_ExactCache");
    }

    #[test]
    fn display_format() {
        let gw = GatewayError::CloudLlm {
            provider: "openai".into(),
            message: "timeout".into(),
            class: ErrorClass::Retryable,
        };
        assert_eq!(format!("{gw}"), "[openai] timeout");
    }

    #[test]
    fn inference_error_classification() {
        let err = anyhow::anyhow!("connection refused");
        let gw = GatewayError::from_inference_error(&err);
        assert!(gw.is_retryable());
        assert_eq!(gw.layer_label(), "L2_SLM");
    }
}

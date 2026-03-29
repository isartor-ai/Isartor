use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Context as AnyhowContext, Result};
use axum::body::{Body, Bytes};
use axum::http::HeaderMap;
use chrono::Utc;
use http_body::Body as HttpBody;
use http_body::Frame;
use serde::Serialize;
use serde_json::Value;

use crate::config::AppConfig;
use crate::core::prompt::extract_request_model;

const REQUEST_LOG_FILE_NAME: &str = "requests.log";
const REQUEST_LOG_MAX_BODY_BYTES: usize = 16 * 1024;
const REQUEST_LOG_MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;
const REQUEST_LOG_ROTATIONS: usize = 5;

#[derive(Debug, Clone)]
pub struct RequestLogContext {
    timestamp: String,
    method: String,
    path: String,
    endpoint_family: String,
    client: String,
    tool: String,
    provider: String,
    model: Option<String>,
    request_headers: BTreeMap<String, String>,
    request_body: String,
    request_body_truncated: bool,
    response_status: u16,
    response_headers: BTreeMap<String, String>,
    latency_ms: u64,
    final_layer: String,
}

#[derive(Debug, Serialize)]
struct RequestLogRecord {
    timestamp: String,
    method: String,
    path: String,
    endpoint_family: String,
    client: String,
    tool: String,
    provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    request_headers: BTreeMap<String, String>,
    request_body: String,
    request_body_truncated: bool,
    response_status: u16,
    response_headers: BTreeMap<String, String>,
    response_body: String,
    response_body_truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_body_note: Option<String>,
    latency_ms: u64,
    final_layer: String,
}

pub struct LoggingBody {
    inner: Body,
    config: Arc<AppConfig>,
    context: RequestLogContext,
    captured: Vec<u8>,
    truncated: bool,
    emitted: bool,
}

pub struct RequestLogExchange<'a> {
    pub config: &'a AppConfig,
    pub method: &'a str,
    pub path: &'a str,
    pub endpoint_family: &'a str,
    pub client: &'a str,
    pub tool: &'a str,
    pub request_headers: &'a HeaderMap,
    pub request_body: &'a [u8],
    pub response_status: u16,
    pub response_headers: &'a HeaderMap,
    pub latency_ms: u64,
    pub final_layer: &'a str,
}

impl RequestLogContext {
    pub fn from_exchange(exchange: RequestLogExchange<'_>) -> Self {
        let (request_body, request_body_truncated) = sanitize_body(exchange.request_body);
        Self {
            timestamp: Utc::now().to_rfc3339(),
            method: exchange.method.to_string(),
            path: exchange.path.to_string(),
            endpoint_family: exchange.endpoint_family.to_string(),
            client: exchange.client.to_string(),
            tool: exchange.tool.to_string(),
            provider: exchange.config.llm_provider.to_string(),
            model: logged_model(
                exchange.config,
                exchange.endpoint_family,
                request_body.as_bytes(),
            ),
            request_headers: redact_headers(exchange.request_headers),
            request_body,
            request_body_truncated,
            response_status: exchange.response_status,
            response_headers: redact_headers(exchange.response_headers),
            latency_ms: exchange.latency_ms,
            final_layer: exchange.final_layer.to_string(),
        }
    }
}

impl LoggingBody {
    pub fn new(inner: Body, config: Arc<AppConfig>, context: RequestLogContext) -> Self {
        Self {
            inner,
            config,
            context,
            captured: Vec::new(),
            truncated: false,
            emitted: false,
        }
    }

    fn capture(&mut self, bytes: &[u8]) {
        if self.truncated || bytes.is_empty() {
            return;
        }

        let remaining = REQUEST_LOG_MAX_BODY_BYTES.saturating_sub(self.captured.len());
        if remaining == 0 {
            self.truncated = true;
            return;
        }

        let take = remaining.min(bytes.len());
        self.captured.extend_from_slice(&bytes[..take]);
        if take < bytes.len() {
            self.truncated = true;
        }
    }

    fn finish(&mut self, note: Option<String>) {
        if self.emitted {
            return;
        }

        let (response_body, sanitized_truncated) = sanitize_body(&self.captured);
        let response_body_truncated = self.truncated || sanitized_truncated;
        if let Err(error) = append_request_log(
            &self.config,
            &self.context,
            response_body,
            response_body_truncated,
            note,
        ) {
            tracing::warn!(error = %error, "Failed to write request log entry");
        }
        self.emitted = true;
    }
}

impl Drop for LoggingBody {
    fn drop(&mut self) {
        if !self.emitted {
            self.finish(Some(
                "response body logging ended before stream completion".to_string(),
            ));
        }
    }
}

impl HttpBody for LoggingBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match Pin::new(&mut self.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    self.capture(data);
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(err))) => {
                self.finish(Some(format!("response stream error: {err}")));
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(None) => {
                self.finish(None);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

pub fn request_log_file_path(config: &AppConfig) -> Result<PathBuf> {
    Ok(expand_request_log_dir(&config.request_log_path)?.join(REQUEST_LOG_FILE_NAME))
}

pub fn configured_request_log_file_path() -> Result<PathBuf> {
    match AppConfig::load_with_validation(false) {
        Ok(config) => request_log_file_path(&config),
        Err(_) => Ok(default_request_log_dir()?.join(REQUEST_LOG_FILE_NAME)),
    }
}

pub fn default_request_log_dir_string() -> String {
    "~/.isartor/request_logs".to_string()
}

pub fn default_request_log_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".isartor").join("request_logs"))
}

fn expand_request_log_dir(path: &str) -> Result<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        anyhow::bail!("request_log_path must not be empty");
    }

    if trimmed == "~" {
        return default_request_log_dir().map(|dir| dir.parent().unwrap_or(&dir).to_path_buf());
    }

    if let Some(rest) = trimmed.strip_prefix("~/") {
        let home = dirs::home_dir().context("cannot determine home directory")?;
        return Ok(home.join(rest));
    }

    Ok(PathBuf::from(trimmed))
}

pub fn logged_model(
    config: &AppConfig,
    endpoint_family: &str,
    request_body: &[u8],
) -> Option<String> {
    extract_request_model(request_body)
        .map(|requested| config.resolve_model_alias(&requested))
        .or_else(|| match endpoint_family {
            "native" | "openai" | "anthropic" => Some(config.configured_model_id()),
            _ => None,
        })
}

fn append_request_log(
    config: &AppConfig,
    context: &RequestLogContext,
    response_body: String,
    response_body_truncated: bool,
    response_body_note: Option<String>,
) -> Result<()> {
    let path = request_log_file_path(config)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create request log directory {}",
            parent.display()
        )
    })?;
    rotate_request_log_if_needed(&path)?;

    let record = RequestLogRecord {
        timestamp: context.timestamp.clone(),
        method: context.method.clone(),
        path: context.path.clone(),
        endpoint_family: context.endpoint_family.clone(),
        client: context.client.clone(),
        tool: context.tool.clone(),
        provider: context.provider.clone(),
        model: context.model.clone(),
        request_headers: context.request_headers.clone(),
        request_body: context.request_body.clone(),
        request_body_truncated: context.request_body_truncated,
        response_status: context.response_status,
        response_headers: context.response_headers.clone(),
        response_body,
        response_body_truncated,
        response_body_note,
        latency_ms: context.latency_ms,
        final_layer: context.final_layer.clone(),
    };

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::to_writer(&mut file, &record)
        .with_context(|| format!("failed to serialize {}", path.display()))?;
    writeln!(file).with_context(|| format!("failed to append newline to {}", path.display()))?;
    Ok(())
}

fn rotate_request_log_if_needed(path: &Path) -> Result<()> {
    let current_size = match fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err).with_context(|| format!("failed to stat {}", path.display())),
    };

    if current_size < REQUEST_LOG_MAX_FILE_BYTES {
        return Ok(());
    }

    let oldest = path.with_extension(format!("log.{REQUEST_LOG_ROTATIONS}"));
    if oldest.exists() {
        fs::remove_file(&oldest)
            .with_context(|| format!("failed to remove {}", oldest.display()))?;
    }

    for idx in (1..REQUEST_LOG_ROTATIONS).rev() {
        let source = path.with_extension(format!("log.{idx}"));
        if source.exists() {
            let target = path.with_extension(format!("log.{}", idx + 1));
            fs::rename(&source, &target).with_context(|| {
                format!(
                    "failed to rotate {} -> {}",
                    source.display(),
                    target.display()
                )
            })?;
        }
    }

    if path.exists() {
        let first = path.with_extension("log.1");
        fs::rename(path, &first).with_context(|| {
            format!("failed to rotate {} -> {}", path.display(), first.display())
        })?;
    }

    Ok(())
}

fn redact_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| {
            let key = name.as_str().to_string();
            let rendered = if should_redact_header(name.as_str()) {
                "[REDACTED]".to_string()
            } else {
                value.to_str().unwrap_or("<non-utf8>").to_string()
            };
            (key, rendered)
        })
        .collect()
}

fn should_redact_header(header_name: &str) -> bool {
    matches!(
        header_name.to_ascii_lowercase().as_str(),
        "authorization" | "proxy-authorization" | "api-key" | "x-api-key" | "cookie" | "set-cookie"
    )
}

fn sanitize_body(bytes: &[u8]) -> (String, bool) {
    if bytes.is_empty() {
        return (String::new(), false);
    }

    let mut redacted = if let Ok(mut json) = serde_json::from_slice::<Value>(bytes) {
        redact_json_value(&mut json);
        serde_json::to_string(&json).unwrap_or_else(|_| String::from_utf8_lossy(bytes).to_string())
    } else {
        String::from_utf8_lossy(bytes).to_string()
    };

    if redacted.len() <= REQUEST_LOG_MAX_BODY_BYTES {
        return (redacted, false);
    }

    let mut truncated = 0usize;
    while !redacted.is_char_boundary(REQUEST_LOG_MAX_BODY_BYTES - truncated) {
        truncated += 1;
    }
    redacted.truncate(REQUEST_LOG_MAX_BODY_BYTES - truncated);
    redacted.push_str("… [truncated]");
    (redacted, true)
}

fn redact_json_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, nested) in map.iter_mut() {
                if should_redact_body_key(key) {
                    *nested = Value::String("[REDACTED]".to_string());
                } else {
                    redact_json_value(nested);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_json_value(item);
            }
        }
        _ => {}
    }
}

fn should_redact_body_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "authorization"
            | "api-key"
            | "x-api-key"
            | "api_key"
            | "apikey"
            | "external_llm_api_key"
            | "provider_keys"
            | "token"
            | "access_token"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;

    fn test_config(path: &str) -> AppConfig {
        let mut cfg = AppConfig::test_default();
        cfg.enable_request_logs = true;
        cfg.request_log_path = path.to_string();
        cfg
    }

    #[test]
    fn sanitize_body_redacts_sensitive_json_keys() {
        let (body, truncated) = sanitize_body(
            br#"{"prompt":"hello","api_key":"secret","nested":{"authorization":"Bearer abc"}}"#,
        );
        assert!(!truncated);
        assert!(body.contains("\"api_key\":\"[REDACTED]\""));
        assert!(body.contains("\"authorization\":\"[REDACTED]\""));
        assert!(!body.contains("secret"));
    }

    #[test]
    fn sanitize_body_redacts_provider_key_arrays() {
        let (body, truncated) =
            sanitize_body(br#"{"provider_keys":[{"key":"sk-secret","label":"team"}]}"#);
        assert!(!truncated);
        assert!(body.contains("\"provider_keys\":\"[REDACTED]\""));
        assert!(!body.contains("sk-secret"));
    }

    #[test]
    fn redact_headers_masks_sensitive_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        headers.insert("x-api-key", "abc123".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());

        let redacted = redact_headers(&headers);
        assert_eq!(redacted.get("authorization").unwrap(), "[REDACTED]");
        assert_eq!(redacted.get("x-api-key").unwrap(), "[REDACTED]");
        assert_eq!(redacted.get("content-type").unwrap(), "application/json");
    }

    #[tokio::test]
    async fn logging_body_writes_jsonl_record() {
        let temp = tempfile::tempdir().unwrap();
        let config = Arc::new(test_config(temp.path().to_str().unwrap()));
        let request_headers = HeaderMap::new();
        let response_headers = HeaderMap::new();
        let context = RequestLogContext::from_exchange(RequestLogExchange {
            config: &config,
            method: "POST",
            path: "/api/chat",
            endpoint_family: "native",
            client: "direct",
            tool: "unknown",
            request_headers: &request_headers,
            request_body: br#"{"prompt":"hello"}"#,
            response_status: 200,
            response_headers: &response_headers,
            latency_ms: 12,
            final_layer: "L3_Cloud",
        });

        let body = Body::new(LoggingBody::new(
            Body::from(r#"{"message":"world"}"#),
            config.clone(),
            context,
        ));
        let collected = body.collect().await.unwrap().to_bytes();
        assert_eq!(collected, Bytes::from_static(br#"{"message":"world"}"#));

        let log_path = request_log_file_path(&config).unwrap();
        let contents = fs::read_to_string(log_path).unwrap();
        assert!(contents.contains("\"path\":\"/api/chat\""));
        assert!(contents.contains("\"request_body\":\"{\\\"prompt\\\":\\\"hello\\\"}\""));
        assert!(contents.contains("\"response_body\":\"{\\\"message\\\":\\\"world\\\"}\""));
    }
}

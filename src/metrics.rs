//! # Firewall Metrics — Lazy-initialised OTel instruments
//!
//! All metric instruments are created once from the global `MeterProvider`
//! and cached for the lifetime of the process.  When monitoring is disabled
//! the instruments are still constructed (they become no-ops because the
//! global provider is the default no-op provider).
//!
//! ## Instruments
//!
//! | Name                                 | Type      | Labels                              |
//! |--------------------------------------|-----------|--------------------------------------|
//! | `isartor_requests_total`             | Counter   | `final_layer`, `status_code`, `traffic_surface`, `client`, `endpoint_family`, `tool` |
//! | `isartor_layer_duration_seconds`     | Histogram | `layer_name`, `tool`                 |
//! | `isartor_tokens_saved_total`         | Counter   | `final_layer`, `traffic_surface`, `client`, `endpoint_family`, `tool` |
//! | `isartor_errors_total`               | Counter   | `layer`, `error_class`, `tool`       |
//! | `isartor_retries_total`              | Counter   | `operation`, `attempts`, `outcome`, `tool` |
//! | `isartor_cache_events_total`         | Counter   | `cache_layer`, `outcome`, `tool`     |

use opentelemetry::metrics::{Counter, Histogram};
use opentelemetry::{KeyValue, global};
use std::sync::OnceLock;

/// Cached set of OTel metric instruments.
pub struct GatewayMetrics {
    /// Total requests processed, labelled by the final handling layer and HTTP status.
    pub requests_total: Counter<u64>,

    /// End-to-end request latency in seconds, labelled by the final layer.
    pub request_duration_seconds: Histogram<f64>,

    /// Per-layer latency in seconds (e.g. cache lookup, SLM inference, LLM call).
    pub layer_duration_seconds: Histogram<f64>,

    /// Cloud tokens we *avoided* paying for because the request was
    /// resolved by an earlier layer (L1a, L1b, or L2).
    /// This is the primary ROI metric for cost-savings dashboards.
    pub tokens_saved_total: Counter<u64>,

    /// Total errors emitted by each layer, labelled by error class (fatal / retryable).
    pub errors_total: Counter<u64>,

    /// Total retry attempts, labelled by operation and outcome (success / exhausted).
    pub retries_total: Counter<u64>,

    /// Cache hits and misses, labelled by cache layer (L1 / L1a / L1b) and tool.
    pub cache_events_total: Counter<u64>,
}

/// Singleton accessor.  The instruments are created on first call.
pub fn metrics() -> &'static GatewayMetrics {
    static INSTANCE: OnceLock<GatewayMetrics> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let meter = global::meter("isartor.gateway");

        GatewayMetrics {
            requests_total: meter
                .u64_counter("isartor_requests_total")
                .with_description("Total requests processed, labelled by final layer and status")
                .build(),

            request_duration_seconds: meter
                .f64_histogram("isartor_request_duration_seconds")
                .with_description("End-to-end request latency in seconds")
                .with_unit("s")
                .build(),

            layer_duration_seconds: meter
                .f64_histogram("isartor_layer_duration_seconds")
                .with_description("Per-layer processing latency in seconds")
                .with_unit("s")
                .build(),

            tokens_saved_total: meter
                .u64_counter("isartor_tokens_saved_total")
                .with_description(
                    "Estimated cloud LLM tokens saved by early resolution (L1a/L1b/L2)",
                )
                .build(),

            errors_total: meter
                .u64_counter("isartor_errors_total")
                .with_description("Total errors emitted by each layer")
                .build(),

            retries_total: meter
                .u64_counter("isartor_retries_total")
                .with_description("Total retry attempts and their outcomes")
                .build(),

            cache_events_total: meter
                .u64_counter("isartor_cache_events_total")
                .with_description("Cache hits and misses by cache layer and tool")
                .build(),
        }
    })
}

// ── Convenience helpers ──────────────────────────────────────────────

/// Estimate prompt tokens using a simple ~4 chars/token heuristic.
pub fn estimate_prompt_tokens(prompt: &str) -> u64 {
    (prompt.chars().count() as u64 / 4).max(1)
}

/// Estimate completion tokens using the same lightweight heuristic.
pub fn estimate_completion_tokens(response: &str) -> u64 {
    (response.chars().count() as u64 / 4).max(1)
}

/// Estimate the number of tokens a prompt would have consumed if sent to
/// a cloud LLM. Uses prompt tokens plus a conservative completion budget.
pub fn estimate_tokens(prompt: &str) -> u64 {
    estimate_prompt_tokens(prompt) + 256
}

fn request_attrs(
    final_layer: &str,
    status_code: u16,
    traffic_surface: &str,
    client: &str,
    endpoint_family: &str,
) -> [KeyValue; 5] {
    [
        KeyValue::new("final_layer", final_layer.to_string()),
        KeyValue::new("status_code", status_code.to_string()),
        KeyValue::new("traffic_surface", traffic_surface.to_string()),
        KeyValue::new("client", client.to_string()),
        KeyValue::new("endpoint_family", endpoint_family.to_string()),
    ]
}

fn request_attrs_with_tool(
    final_layer: &str,
    status_code: u16,
    traffic_surface: &str,
    client: &str,
    endpoint_family: &str,
    tool: &str,
) -> [KeyValue; 6] {
    [
        KeyValue::new("final_layer", final_layer.to_string()),
        KeyValue::new("status_code", status_code.to_string()),
        KeyValue::new("traffic_surface", traffic_surface.to_string()),
        KeyValue::new("client", client.to_string()),
        KeyValue::new("endpoint_family", endpoint_family.to_string()),
        KeyValue::new("tool", tool.to_string()),
    ]
}

fn layer_duration_attrs(layer_name: &str, tool: &str) -> [KeyValue; 2] {
    [
        KeyValue::new("layer_name", layer_name.to_string()),
        KeyValue::new("tool", tool.to_string()),
    ]
}

fn error_attrs(layer: &str, error_class: &str, tool: &str) -> [KeyValue; 3] {
    [
        KeyValue::new("layer", layer.to_string()),
        KeyValue::new("error_class", error_class.to_string()),
        KeyValue::new("tool", tool.to_string()),
    ]
}

fn retry_attrs(operation: &str, attempts: u32, succeeded: bool, tool: &str) -> [KeyValue; 4] {
    [
        KeyValue::new("operation", operation.to_string()),
        KeyValue::new("attempts", attempts.to_string()),
        KeyValue::new(
            "outcome",
            if succeeded { "success" } else { "exhausted" }.to_string(),
        ),
        KeyValue::new("tool", tool.to_string()),
    ]
}

fn cache_event_attrs(cache_layer: &str, outcome: &str, tool: &str) -> [KeyValue; 3] {
    [
        KeyValue::new("cache_layer", cache_layer.to_string()),
        KeyValue::new("outcome", outcome.to_string()),
        KeyValue::new("tool", tool.to_string()),
    ]
}

/// Record a request completion against the global metrics.
pub fn record_request(final_layer: &str, status_code: u16, duration_secs: f64) {
    record_request_with_context(
        final_layer,
        status_code,
        duration_secs,
        "gateway",
        "direct",
        "native",
    );
}

/// Record a request completion with additional request-surface dimensions.
pub fn record_request_with_context(
    final_layer: &str,
    status_code: u16,
    duration_secs: f64,
    traffic_surface: &str,
    client: &str,
    endpoint_family: &str,
) {
    let m = metrics();
    let attrs = request_attrs(
        final_layer,
        status_code,
        traffic_surface,
        client,
        endpoint_family,
    );
    m.requests_total.add(1, &attrs);
    m.request_duration_seconds.record(duration_secs, &attrs);
}

/// Record a request with tool identification (the preferred function).
pub fn record_request_with_tool(
    final_layer: &str,
    status_code: u16,
    duration_secs: f64,
    traffic_surface: &str,
    client: &str,
    endpoint_family: &str,
    tool: &str,
) {
    let m = metrics();
    let attrs = request_attrs_with_tool(
        final_layer,
        status_code,
        traffic_surface,
        client,
        endpoint_family,
        tool,
    );
    m.requests_total.add(1, &attrs);
    m.request_duration_seconds.record(duration_secs, &attrs);
}

/// Record per-layer latency.
pub fn record_layer_duration(layer_name: &str, duration: std::time::Duration) {
    record_layer_duration_with_tool(layer_name, duration, "unknown");
}

/// Record per-layer latency with tool identification.
pub fn record_layer_duration_with_tool(
    layer_name: &str,
    duration: std::time::Duration,
    tool: &str,
) {
    let m = metrics();
    let attrs = layer_duration_attrs(layer_name, tool);
    m.layer_duration_seconds
        .record(duration.as_secs_f64(), &attrs);
}

/// Record tokens saved (call when a request is resolved before Layer 3).
pub fn record_tokens_saved(final_layer: &str, estimated_tokens: u64) {
    record_tokens_saved_with_context(final_layer, estimated_tokens, "gateway", "direct", "native");
}

/// Record tokens saved with additional request-surface dimensions.
pub fn record_tokens_saved_with_context(
    final_layer: &str,
    estimated_tokens: u64,
    traffic_surface: &str,
    client: &str,
    endpoint_family: &str,
) {
    let m = metrics();
    let attrs = [
        KeyValue::new("final_layer", final_layer.to_string()),
        KeyValue::new("traffic_surface", traffic_surface.to_string()),
        KeyValue::new("client", client.to_string()),
        KeyValue::new("endpoint_family", endpoint_family.to_string()),
    ];
    m.tokens_saved_total.add(estimated_tokens, &attrs);
}

/// Record tokens saved with tool identification (the preferred function).
pub fn record_tokens_saved_with_tool(
    final_layer: &str,
    estimated_tokens: u64,
    traffic_surface: &str,
    client: &str,
    endpoint_family: &str,
    tool: &str,
) {
    let m = metrics();
    let attrs = [
        KeyValue::new("final_layer", final_layer.to_string()),
        KeyValue::new("traffic_surface", traffic_surface.to_string()),
        KeyValue::new("client", client.to_string()),
        KeyValue::new("endpoint_family", endpoint_family.to_string()),
        KeyValue::new("tool", tool.to_string()),
    ];
    m.tokens_saved_total.add(estimated_tokens, &attrs);
}

/// Record an error occurrence, labelled by the layer that produced it and
/// the error class (`fatal` or `retryable`).
pub fn record_error(layer: &str, error_class: &str) {
    record_error_with_tool(layer, error_class, "unknown");
}

/// Record an error occurrence with tool identification.
pub fn record_error_with_tool(layer: &str, error_class: &str, tool: &str) {
    let m = metrics();
    let attrs = error_attrs(layer, error_class, tool);
    m.errors_total.add(1, &attrs);
}

/// Record a retry event, labelled by the operation name and outcome
/// (`success` or `exhausted`).
pub fn record_retry(operation: &str, attempts: u32, succeeded: bool) {
    record_retry_with_tool(operation, attempts, succeeded, "unknown");
}

/// Record a retry event with tool identification.
pub fn record_retry_with_tool(operation: &str, attempts: u32, succeeded: bool, tool: &str) {
    let m = metrics();
    let attrs = retry_attrs(operation, attempts, succeeded, tool);
    m.retries_total.add(1, &attrs);
}

/// Record a cache hit or miss with tool identification.
pub fn record_cache_event_with_tool(cache_layer: &str, outcome: &str, tool: &str) {
    let m = metrics();
    let attrs = cache_event_attrs(cache_layer, outcome, tool);
    m.cache_events_total.add(1, &attrs);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_value<'a>(attrs: &'a [KeyValue], key: &str) -> Option<&'a str> {
        attrs.iter().find_map(|attr| {
            if attr.key.as_str() == key {
                match &attr.value {
                    opentelemetry::Value::String(value) => Some(value.as_str()),
                    _ => None,
                }
            } else {
                None
            }
        })
    }

    #[test]
    fn tool_label_is_added_to_request_metrics() {
        let attrs = request_attrs_with_tool("l1a", 200, "gateway", "direct", "native", "cursor");
        assert_eq!(key_value(&attrs, "tool"), Some("cursor"));
        assert_eq!(key_value(&attrs, "final_layer"), Some("l1a"));
    }

    #[test]
    fn tool_label_is_added_to_error_retry_and_cache_metrics() {
        let error = error_attrs("L3_Cloud", "fatal", "copilot");
        assert_eq!(key_value(&error, "tool"), Some("copilot"));

        let retry = retry_attrs("L3_Cloud_LLM", 3, false, "copilot");
        assert_eq!(key_value(&retry, "tool"), Some("copilot"));
        assert_eq!(key_value(&retry, "outcome"), Some("exhausted"));

        let cache = cache_event_attrs("l1a", "miss", "copilot");
        assert_eq!(key_value(&cache, "tool"), Some("copilot"));
        assert_eq!(key_value(&cache, "cache_layer"), Some("l1a"));
    }
}

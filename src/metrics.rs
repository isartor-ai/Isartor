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
//! | `isartor_requests_total`             | Counter   | `final_layer`, `status_code`         |
//! | `isartor_layer_duration_seconds`     | Histogram | `layer_name`                         |
//! | `isartor_tokens_saved_total`         | Counter   | `final_layer`                        |
//! | `isartor_errors_total`               | Counter   | `layer`, `error_class`               |
//! | `isartor_retries_total`              | Counter   | `operation`, `outcome`               |

use opentelemetry::metrics::{Counter, Histogram};
use opentelemetry::{global, KeyValue};
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
        }
    })
}

// ── Convenience helpers ──────────────────────────────────────────────

/// Estimate the number of tokens a prompt would have consumed if sent to
/// a cloud LLM.  Uses the simple heuristic of ~4 characters per token
/// (GPT-style BPE).  A more accurate approach would use `tiktoken`, but
/// the rough estimate is sufficient for cost dashboards.
pub fn estimate_tokens(prompt: &str) -> u64 {
    // prompt tokens + a modest estimate for completion tokens
    let prompt_tokens = (prompt.len() as u64) / 4;
    let completion_estimate = 256; // conservative average
    prompt_tokens + completion_estimate
}

/// Record a request completion against the global metrics.
pub fn record_request(final_layer: &str, status_code: u16, duration_secs: f64) {
    let m = metrics();
    let attrs = [
        KeyValue::new("final_layer", final_layer.to_string()),
        KeyValue::new("status_code", status_code.to_string()),
    ];
    m.requests_total.add(1, &attrs);
    m.request_duration_seconds.record(duration_secs, &attrs);
}

/// Record per-layer latency.
pub fn record_layer_duration(layer_name: &str, duration: std::time::Duration) {
    let m = metrics();
    m.layer_duration_seconds.record(
        duration.as_secs_f64(),
        &[KeyValue::new("layer_name", layer_name.to_string())],
    );
}

/// Record tokens saved (call when a request is resolved before Layer 3).
pub fn record_tokens_saved(final_layer: &str, estimated_tokens: u64) {
    let m = metrics();
    m.tokens_saved_total.add(
        estimated_tokens,
        &[KeyValue::new("final_layer", final_layer.to_string())],
    );
}

/// Record an error occurrence, labelled by the layer that produced it and
/// the error class (`fatal` or `retryable`).
pub fn record_error(layer: &str, error_class: &str) {
    let m = metrics();
    m.errors_total.add(
        1,
        &[
            KeyValue::new("layer", layer.to_string()),
            KeyValue::new("error_class", error_class.to_string()),
        ],
    );
}

/// Record a retry event, labelled by the operation name and outcome
/// (`success` or `exhausted`).
pub fn record_retry(operation: &str, attempts: u32, succeeded: bool) {
    let m = metrics();
    m.retries_total.add(
        1,
        &[
            KeyValue::new("operation", operation.to_string()),
            KeyValue::new("attempts", attempts.to_string()),
            KeyValue::new(
                "outcome",
                if succeeded { "success" } else { "exhausted" }.to_string(),
            ),
        ],
    );
}

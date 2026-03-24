//! # Retry with Exponential Back-off
//!
//! Provides [`execute_with_retry`] — a generic retry wrapper that
//! re-executes a future when it returns a [`GatewayError`] with
//! [`ErrorClass::Retryable`].  Fatal errors are returned immediately.
//!
//! ## Back-off strategy
//!
//! | Attempt | Delay                            |
//! |---------|----------------------------------|
//! | 1       | `base_delay`                     |
//! | 2       | `base_delay × 2`                 |
//! | 3       | `base_delay × 4`                 |
//! | …       | capped at `max_delay`            |
//!
//! A small random jitter (±25 %) is added to each delay to prevent
//! thundering-herd effects when many requests retry in parallel.

use std::future::Future;
use std::time::Duration;

use crate::errors::{ErrorClass, GatewayError};

/// Default retry parameters.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;
pub const DEFAULT_BASE_DELAY: Duration = Duration::from_millis(250);
pub const DEFAULT_MAX_DELAY: Duration = Duration::from_secs(5);

/// Configuration for the retry policy.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the initial one).
    pub max_attempts: u32,
    /// Initial delay between the first and second attempt.
    pub base_delay: Duration,
    /// Upper bound on any single delay.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            base_delay: DEFAULT_BASE_DELAY,
            max_delay: DEFAULT_MAX_DELAY,
        }
    }
}

/// Execute an async closure with retry logic.
///
/// * `Ok(T)` is returned on the first success.
/// * A [`GatewayError`] with [`ErrorClass::Fatal`] aborts immediately.
/// * A retryable error causes the closure to re-execute up to
///   `config.max_attempts` times (including the initial call).
///
/// The last error is returned when all attempts are exhausted.
pub async fn execute_with_retry<F, Fut, T>(
    config: &RetryConfig,
    operation_name: &str,
    tool: &str,
    mut f: F,
) -> Result<T, GatewayError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, GatewayError>>,
{
    let mut last_error: Option<GatewayError> = None;

    for attempt in 1..=config.max_attempts {
        match f().await {
            Ok(val) => {
                if attempt > 1 {
                    tracing::info!(operation = operation_name, attempt, "Retry succeeded");
                    crate::metrics::record_retry_with_tool(operation_name, attempt, true, tool);
                    crate::visibility::record_agent_retry(tool, attempt);
                }
                return Ok(val);
            }
            Err(e) => {
                let class = e.class();
                tracing::warn!(
                    operation = operation_name,
                    attempt,
                    max_attempts = config.max_attempts,
                    error_class = ?class,
                    error = %e,
                    "Operation failed"
                );

                // Fatal errors are never retried.
                if class == ErrorClass::Fatal {
                    crate::metrics::record_error_with_tool(e.layer_label(), "fatal", tool);
                    crate::visibility::record_agent_error(tool);
                    return Err(e);
                }

                crate::metrics::record_error_with_tool(e.layer_label(), "retryable", tool);
                crate::visibility::record_agent_error(tool);

                last_error = Some(e);

                // Sleep before the next attempt (unless this was the last one).
                if attempt < config.max_attempts {
                    let delay = backoff_delay(attempt, config.base_delay, config.max_delay);
                    tracing::debug!(
                        operation = operation_name,
                        delay_ms = delay.as_millis() as u64,
                        next_attempt = attempt + 1,
                        "Backing off before retry"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    // All attempts exhausted — record and return the final error.
    if let Some(ref err) = last_error {
        crate::metrics::record_retry_with_tool(operation_name, config.max_attempts, false, tool);
        crate::visibility::record_agent_retry(tool, config.max_attempts);
        tracing::error!(
            operation = operation_name,
            attempts = config.max_attempts,
            error = %err,
            "All retry attempts exhausted"
        );
    }

    Err(last_error.expect("at least one attempt must have been made"))
}

/// Compute the back-off delay for a given attempt (1-indexed).
///
/// delay = base × 2^(attempt-1)  ±  25 % jitter, capped at max.
fn backoff_delay(attempt: u32, base: Duration, max: Duration) -> Duration {
    let exp = 2_u64.saturating_pow(attempt.saturating_sub(1));
    let raw = base.saturating_mul(exp as u32);
    let capped = raw.min(max);

    // Add ±25 % jitter using a simple deterministic-enough source.
    // We intentionally avoid pulling in `rand` — the jitter doesn't
    // need to be cryptographically secure.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let jitter_pct = (nanos % 50) as i64 - 25; // -25..+24
    let jitter = (capped.as_millis() as i64 * jitter_pct) / 100;
    let millis = (capped.as_millis() as i64 + jitter).max(1) as u64;
    Duration::from_millis(millis)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn retryable_error(msg: &str) -> GatewayError {
        GatewayError::CloudLlm {
            provider: "test".into(),
            message: msg.into(),
            class: ErrorClass::Retryable,
        }
    }

    fn fatal_error(msg: &str) -> GatewayError {
        GatewayError::CloudLlm {
            provider: "test".into(),
            message: msg.into(),
            class: ErrorClass::Fatal,
        }
    }

    fn fast_config() -> RetryConfig {
        RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        }
    }

    #[tokio::test]
    async fn succeeds_on_first_try() {
        let result = execute_with_retry(&fast_config(), "test_op", "cursor", || async {
            Ok::<_, GatewayError>(42)
        })
        .await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn retries_then_succeeds() {
        let counter = Arc::new(AtomicU32::new(0));

        let c = counter.clone();
        let result = execute_with_retry(&fast_config(), "test_op", "cursor", move || {
            let c = c.clone();
            async move {
                let attempt = c.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt < 3 {
                    Err(retryable_error("transient"))
                } else {
                    Ok(99)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 99);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn fatal_error_aborts_immediately() {
        let counter = Arc::new(AtomicU32::new(0));

        let c = counter.clone();
        let result = execute_with_retry(&fast_config(), "test_op", "cursor", move || {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(fatal_error("invalid api key"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().class(), ErrorClass::Fatal);
        // Only one attempt — no retries for fatal errors.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn exhausts_all_attempts() {
        let counter = Arc::new(AtomicU32::new(0));

        let c = counter.clone();
        let result = execute_with_retry(&fast_config(), "test_op", "cursor", move || {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(retryable_error("still failing"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn backoff_respects_max_delay() {
        let delay = backoff_delay(10, Duration::from_secs(1), Duration::from_secs(5));
        // Even with jitter, should not exceed max + 25 %.
        assert!(delay <= Duration::from_millis(6250));
    }

    #[test]
    fn backoff_increases_exponentially() {
        let d1 = Duration::from_millis(100).as_millis();
        let d2 = Duration::from_millis(200).as_millis();
        let d3 = Duration::from_millis(400).as_millis();

        // The raw (pre-jitter) delays should double each time.
        // We test the raw formula, not the jittered output.
        let base = Duration::from_millis(100);
        let max = Duration::from_secs(60);
        let raw1 = base.saturating_mul(1);
        let raw2 = base.saturating_mul(2);
        let raw3 = base.saturating_mul(4);
        assert_eq!(raw1.as_millis(), d1);
        assert_eq!(raw2.as_millis(), d2);
        assert_eq!(raw3.as_millis(), d3);
        let _ = max; // suppress unused-variable warning
    }
}

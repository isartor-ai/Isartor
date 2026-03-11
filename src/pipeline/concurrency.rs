// =============================================================================
// Adaptive Concurrency Limiter — Layer 0 (Ops)
//
// Protects the pipeline from overload by tracking recent request latencies
// and rejecting new requests when the system is saturated. This implements
// a simplified version of the "adaptive concurrency" pattern used in
// production service meshes (e.g., Envoy, Linkerd).
//
// The algorithm:
// 1. Maintains a sliding window of recent request latencies.
// 2. Computes the P95 latency over the window.
// 3. If current in-flight requests exceed the calculated concurrency
//    limit (derived from Little's Law), new requests are shed.
// =============================================================================

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

/// Configuration for the adaptive concurrency limiter.
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Minimum concurrency limit (floor). The limiter will never drop
    /// below this, even if latencies are very high.
    pub min_concurrency: u64,

    /// Maximum concurrency limit (ceiling). Safety valve.
    pub max_concurrency: u64,

    /// Target latency. If P95 latency exceeds this, the concurrency
    /// limit is reduced.
    pub target_latency: Duration,

    /// Number of latency samples to keep in the sliding window.
    pub window_size: usize,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            min_concurrency: 4,
            max_concurrency: 256,
            target_latency: Duration::from_millis(500),
            window_size: 100,
        }
    }
}

/// Adaptive concurrency limiter based on recent latency observations.
///
/// Thread-safe: uses atomics for in-flight counter and a RwLock for
/// the latency window (writes are infrequent relative to reads).
pub struct AdaptiveConcurrencyLimiter {
    config: ConcurrencyConfig,

    /// Current number of in-flight requests.
    in_flight: AtomicU64,

    /// Current computed concurrency limit.
    concurrency_limit: AtomicU64,

    /// Sliding window of recent request durations for P95 estimation.
    latency_window: RwLock<VecDeque<Duration>>,
}

impl AdaptiveConcurrencyLimiter {
    /// Create a new limiter with the given configuration.
    pub fn new(config: ConcurrencyConfig) -> Self {
        let initial_limit = config.max_concurrency;
        Self {
            config,
            in_flight: AtomicU64::new(0),
            concurrency_limit: AtomicU64::new(initial_limit),
            latency_window: RwLock::new(VecDeque::new()),
        }
    }

    /// Attempt to acquire a permit to proceed through the pipeline.
    ///
    /// Returns `Ok(ConcurrencyPermit)` if the request is admitted, or
    /// `Err(())` if the system is overloaded and the request should be shed.
    ///
    /// The permit automatically decrements the in-flight counter and
    /// records the request latency when dropped.
    #[allow(clippy::result_unit_err)]
    pub fn try_acquire(&self) -> Result<ConcurrencyPermit<'_>, ()> {
        let current = self.in_flight.fetch_add(1, Ordering::AcqRel);
        let limit = self.concurrency_limit.load(Ordering::Acquire);

        if current >= limit {
            // Over the limit — reject and undo the increment.
            self.in_flight.fetch_sub(1, Ordering::Release);
            return Err(());
        }

        Ok(ConcurrencyPermit {
            limiter: self,
            start: Instant::now(),
        })
    }

    /// Current number of in-flight requests (for observability).
    pub fn current_in_flight(&self) -> u64 {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Current computed concurrency limit (for observability).
    pub fn current_limit(&self) -> u64 {
        self.concurrency_limit.load(Ordering::Relaxed)
    }

    /// Record a completed request's latency and recalculate the limit.
    async fn record_latency(&self, duration: Duration) {
        let mut window = self.latency_window.write().await;
        window.push_back(duration);

        // Trim to window size.
        while window.len() > self.config.window_size {
            window.pop_front();
        }

        // Recalculate concurrency limit based on P95.
        if window.len() >= 10 {
            let p95 = self.percentile(&window, 95);

            let new_limit = if p95 > self.config.target_latency {
                // Latencies are too high — reduce concurrency.
                let current = self.concurrency_limit.load(Ordering::Relaxed);
                (current * 9 / 10).max(self.config.min_concurrency) // -10%
            } else {
                // Latencies are healthy — cautiously increase.
                let current = self.concurrency_limit.load(Ordering::Relaxed);
                (current + 1).min(self.config.max_concurrency) // +1
            };

            self.concurrency_limit.store(new_limit, Ordering::Release);

            tracing::debug!(
                p95_ms = p95.as_millis(),
                new_limit = new_limit,
                window_size = window.len(),
                "Adaptive concurrency: limit recalculated"
            );
        }
    }

    /// Compute the given percentile from a window of durations.
    fn percentile(&self, window: &VecDeque<Duration>, pct: usize) -> Duration {
        let mut sorted: Vec<Duration> = window.iter().copied().collect();
        sorted.sort();
        let idx = (sorted.len() * pct / 100)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        sorted[idx]
    }
}

/// RAII permit that tracks an in-flight request.
///
/// When dropped, it decrements the in-flight counter and records
/// the request's latency for the adaptive algorithm.
pub struct ConcurrencyPermit<'a> {
    limiter: &'a AdaptiveConcurrencyLimiter,
    start: Instant,
}

impl<'a> ConcurrencyPermit<'a> {
    /// Explicitly release the permit and record latency.
    ///
    /// This is preferred over relying on `Drop` when you need the
    /// async latency recording to happen.
    pub async fn release(self) {
        let duration = self.start.elapsed();
        self.limiter.in_flight.fetch_sub(1, Ordering::Release);
        self.limiter.record_latency(duration).await;
        // Prevent Drop from double-decrementing.
        std::mem::forget(self);
    }
}

impl<'a> Drop for ConcurrencyPermit<'a> {
    fn drop(&mut self) {
        // Synchronous fallback: decrements in-flight but cannot record
        // latency (requires async). Use `release()` when possible.
        self.limiter.in_flight.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ConcurrencyConfig {
        ConcurrencyConfig {
            min_concurrency: 2,
            max_concurrency: 10,
            target_latency: Duration::from_millis(100),
            window_size: 20,
        }
    }

    // ── ConcurrencyConfig ────────────────────────────────────────

    #[test]
    fn default_config() {
        let cfg = ConcurrencyConfig::default();
        assert_eq!(cfg.min_concurrency, 4);
        assert_eq!(cfg.max_concurrency, 256);
        assert_eq!(cfg.target_latency, Duration::from_millis(500));
        assert_eq!(cfg.window_size, 100);
    }

    // ── AdaptiveConcurrencyLimiter ───────────────────────────────

    #[test]
    fn limiter_initial_state() {
        let limiter = AdaptiveConcurrencyLimiter::new(test_config());
        assert_eq!(limiter.current_in_flight(), 0);
        assert_eq!(limiter.current_limit(), 10); // starts at max
    }

    #[test]
    fn limiter_acquire_and_drop() {
        let limiter = AdaptiveConcurrencyLimiter::new(test_config());
        {
            let permit = limiter.try_acquire().unwrap();
            assert_eq!(limiter.current_in_flight(), 1);
            drop(permit);
        }
        assert_eq!(limiter.current_in_flight(), 0);
    }

    #[test]
    fn limiter_load_shedding() {
        let cfg = ConcurrencyConfig {
            min_concurrency: 1,
            max_concurrency: 2,
            target_latency: Duration::from_millis(100),
            window_size: 20,
        };
        let limiter = AdaptiveConcurrencyLimiter::new(cfg);

        let _p1 = limiter.try_acquire().unwrap();
        let _p2 = limiter.try_acquire().unwrap();
        // Third request should be rejected.
        let result = limiter.try_acquire();
        assert!(result.is_err());
        assert_eq!(limiter.current_in_flight(), 2);
    }

    #[tokio::test]
    async fn limiter_release_records_latency() {
        let limiter = AdaptiveConcurrencyLimiter::new(test_config());
        let permit = limiter.try_acquire().unwrap();
        assert_eq!(limiter.current_in_flight(), 1);

        permit.release().await;
        assert_eq!(limiter.current_in_flight(), 0);

        let window = limiter.latency_window.read().await;
        assert_eq!(window.len(), 1);
    }

    #[tokio::test]
    async fn limiter_limit_decreases_under_high_latency() {
        let cfg = ConcurrencyConfig {
            min_concurrency: 2,
            max_concurrency: 100,
            target_latency: Duration::from_millis(10),
            window_size: 20,
        };
        let limiter = AdaptiveConcurrencyLimiter::new(cfg);

        // Simulate 15 high-latency requests (> target latency).
        for _ in 0..15 {
            limiter.record_latency(Duration::from_millis(500)).await;
        }

        let limit = limiter.current_limit();
        assert!(limit < 100, "Limit should decrease: got {limit}");
    }

    #[tokio::test]
    async fn limiter_limit_increases_under_low_latency() {
        let cfg = ConcurrencyConfig {
            min_concurrency: 2,
            max_concurrency: 100,
            target_latency: Duration::from_millis(500),
            window_size: 20,
        };
        let limiter = AdaptiveConcurrencyLimiter::new(cfg);

        // Record 15 low-latency observations.
        for _ in 0..15 {
            limiter.record_latency(Duration::from_millis(1)).await;
        }

        // The limit should stay at max or increase to max.
        let limit = limiter.current_limit();
        assert_eq!(limit, 100);
    }

    #[tokio::test]
    async fn limiter_limit_never_below_min() {
        let cfg = ConcurrencyConfig {
            min_concurrency: 5,
            max_concurrency: 10,
            target_latency: Duration::from_millis(1),
            window_size: 20,
        };
        let limiter = AdaptiveConcurrencyLimiter::new(cfg);

        // Keep hammering with high latencies.
        for _ in 0..100 {
            limiter.record_latency(Duration::from_secs(10)).await;
        }

        assert!(
            limiter.current_limit() >= 5,
            "Limit must never drop below min_concurrency"
        );
    }

    #[test]
    fn limiter_percentile_calculation() {
        let limiter = AdaptiveConcurrencyLimiter::new(test_config());
        let mut window = VecDeque::new();
        for i in 1..=100 {
            window.push_back(Duration::from_millis(i));
        }
        let p95 = limiter.percentile(&window, 95);
        assert_eq!(p95, Duration::from_millis(95));
    }

    #[test]
    fn limiter_concurrent_acquire_drop() {
        let limiter = AdaptiveConcurrencyLimiter::new(test_config());
        let mut permits = Vec::new();
        for _ in 0..10 {
            permits.push(limiter.try_acquire().unwrap());
        }
        assert_eq!(limiter.current_in_flight(), 10);

        // 11th should fail (limit=10).
        assert!(limiter.try_acquire().is_err());

        drop(permits);
        assert_eq!(limiter.current_in_flight(), 0);
    }

    #[tokio::test]
    async fn limiter_window_trimming() {
        let cfg = ConcurrencyConfig {
            min_concurrency: 1,
            max_concurrency: 100,
            target_latency: Duration::from_millis(500),
            window_size: 5,
        };
        let limiter = AdaptiveConcurrencyLimiter::new(cfg);

        // Record 10 latencies — window should trim to 5.
        for i in 0..10 {
            limiter.record_latency(Duration::from_millis(i)).await;
        }

        let window = limiter.latency_window.read().await;
        assert_eq!(window.len(), 5);
    }
}

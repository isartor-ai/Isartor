// =============================================================================
// tests/scenarios/performance_tests.rs
//
// Performance-oriented scenario tests:
//   - Cache-hit latency stays below threshold
//   - Concurrent throughput under load
// =============================================================================

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use isartor::layer1::layer1a_cache::ExactMatchCache;
use isartor::vector_cache::VectorCache;

// ═══════════════════════════════════════════════════════════════════════
// Exact Cache Latency
// ═══════════════════════════════════════════════════════════════════════

/// A single exact-cache get on a populated cache should complete in < 1 ms.
#[test]
fn exact_cache_hit_latency_under_1ms() {
    let cache = ExactMatchCache::new(NonZeroUsize::new(1000).unwrap());

    // Pre-populate
    for i in 0..500 {
        cache.put(format!("key-{i}"), format!("value-{i}"));
    }

    // Measure 1 000 random reads and take the p99 latency.
    let mut durations = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let key = format!("key-{}", rand_idx(500));
        let start = Instant::now();
        let _ = cache.get(&key);
        durations.push(start.elapsed());
    }
    durations.sort();
    let p99 = durations[989]; // index 989 = 99th percentile of 1000 samples
    eprintln!("ExactMatchCache p99 get latency: {p99:?}");
    assert!(
        p99.as_millis() < 1,
        "p99 cache-hit latency should be < 1 ms, was {p99:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Vector Cache Latency
// ═══════════════════════════════════════════════════════════════════════

/// A single vector-cache search should complete in < 50 ms for 100 entries.
#[tokio::test]
async fn vector_cache_search_latency_under_50ms() {
    let cache = VectorCache::new(0.7, 300, 200);

    // Insert 100 random-ish vectors (dimension 64).
    for i in 0..100 {
        let v: Vec<f32> = (0..64).map(|j| ((i * 7 + j) as f32).sin()).collect();
        cache.insert(v, format!("entry-{i}"), None).await;
    }

    let query: Vec<f32> = (0..64).map(|j| (j as f32 * 0.1).cos()).collect();

    let mut durations = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        let _ = cache.search(&query, None).await;
        durations.push(start.elapsed());
    }
    durations.sort();
    let p99 = durations[98];
    eprintln!("VectorCache p99 search latency (100 entries): {p99:?}");
    assert!(
        p99.as_millis() < 50,
        "p99 vector search latency should be < 50 ms, was {p99:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Concurrent Throughput
// ═══════════════════════════════════════════════════════════════════════

/// 100 concurrent exact-cache reads should all complete within 100 ms total.
#[tokio::test]
async fn exact_cache_concurrent_throughput() {
    let cache = Arc::new(ExactMatchCache::new(NonZeroUsize::new(500).unwrap()));
    for i in 0..200 {
        cache.put(format!("k-{i}"), format!("v-{i}"));
    }

    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..100 {
        let c = cache.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..50 {
                let _ = c.get(&format!("k-{}", rand_idx(200)));
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    let elapsed = start.elapsed();
    eprintln!("100×50 concurrent exact-cache reads: {elapsed:?}");
    assert!(
        elapsed.as_millis() < 100,
        "Concurrent reads should complete within 100 ms, took {elapsed:?}"
    );
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Simple deterministic "random" index generator (no external dep needed).
fn rand_idx(max: usize) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut h = DefaultHasher::new();
    SystemTime::now().hash(&mut h);
    std::thread::current().id().hash(&mut h);
    (h.finish() as usize) % max
}

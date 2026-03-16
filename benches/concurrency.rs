// =============================================================================
// benches/concurrency.rs — Criterion benchmarks for concurrent workloads.
//
// Run with:   cargo bench --bench concurrency
// =============================================================================

use std::num::NonZeroUsize;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use isartor::layer1::layer1a_cache::ExactMatchCache;

// ── Concurrent ExactMatchCache access ────────────────────────────────────

fn bench_exact_cache_concurrent_reads(c: &mut Criterion) {
    let cache = Arc::new(ExactMatchCache::new(NonZeroUsize::new(10_000).unwrap()));

    // Pre-populate
    for i in 0..1_000 {
        cache.put(format!("key-{i}"), format!("value-{i}"));
    }

    let mut group = c.benchmark_group("concurrent_exact_cache_reads");

    for num_threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let mut handles = vec![];
                        for thread_id in 0..num_threads {
                            let cache_clone = Arc::clone(&cache);
                            let handle = tokio::spawn(async move {
                                for i in 0..100 {
                                    let key = format!("key-{}", (thread_id * 100 + i) % 1_000);
                                    black_box(cache_clone.get(&key));
                                }
                            });
                            handles.push(handle);
                        }
                        for handle in handles {
                            let _ = handle.await;
                        }
                    });
            },
        );
    }
    group.finish();
}

fn bench_exact_cache_concurrent_writes(c: &mut Criterion) {
    let cache = Arc::new(ExactMatchCache::new(NonZeroUsize::new(100_000).unwrap()));

    let mut group = c.benchmark_group("concurrent_exact_cache_writes");

    for num_threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let mut handles = vec![];

                        for thread_id in 0..num_threads {
                            let cache_clone = Arc::clone(&cache);
                            let handle = tokio::spawn(async move {
                                for i in 0..10 {
                                    let idx = thread_id * 10 + i;
                                    cache_clone.put(format!("key-{idx}"), format!("value-{idx}"));
                                }
                            });
                            handles.push(handle);
                        }
                        for handle in handles {
                            let _ = handle.await;
                        }
                    });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_exact_cache_concurrent_reads,
    bench_exact_cache_concurrent_writes
);
criterion_main!(benches);

// =============================================================================
// benches/cache_latency.rs — Criterion benchmarks for cache subsystem.
//
// Run with:   cargo bench --bench cache_latency
// =============================================================================

use std::num::NonZeroUsize;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use isartor::layer1::layer1a_cache::ExactMatchCache;
use isartor::vector_cache::VectorCache;

// ── ExactMatchCache ──────────────────────────────────────────────────

fn bench_exact_cache_insert(c: &mut Criterion) {
    let cache = ExactMatchCache::new(NonZeroUsize::new(10_000).unwrap());

    c.bench_function("exact_cache_insert", |b| {
        let mut i = 0u64;
        b.iter(|| {
            cache.put(format!("key-{i}"), format!("value-{i}"));
            i += 1;
        })
    });
}

fn bench_exact_cache_get_hit(c: &mut Criterion) {
    let cache = ExactMatchCache::new(NonZeroUsize::new(10_000).unwrap());

    // Pre-populate
    for i in 0..5_000 {
        cache.put(format!("key-{i}"), format!("value-{i}"));
    }

    c.bench_function("exact_cache_get_hit", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let key = format!("key-{}", i % 5_000);
            black_box(cache.get(&key));
            i += 1;
        })
    });
}

fn bench_exact_cache_get_miss(c: &mut Criterion) {
    let cache = ExactMatchCache::new(NonZeroUsize::new(1_000).unwrap());

    c.bench_function("exact_cache_get_miss", |b| {
        let mut i = 0u64;
        b.iter(|| {
            black_box(cache.get(&format!("miss-{i}")));
            i += 1;
        })
    });
}

// ── VectorCache ──────────────────────────────────────────────────────

fn bench_vector_cache_insert(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cache = VectorCache::new(0.85, 3600, 10_000);

    c.bench_function("vector_cache_insert", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let v: Vec<f32> = (0..64).map(|j| ((i * 7 + j) as f32).sin()).collect();
            rt.block_on(cache.insert(v, format!("entry-{i}"), None));
            i += 1;
        })
    });
}

fn bench_vector_cache_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cache = VectorCache::new(0.85, 3600, 10_000);

    // Pre-populate with 500 vectors
    rt.block_on(async {
        for i in 0..500 {
            let v: Vec<f32> = (0..64).map(|j| ((i * 7 + j) as f32).sin()).collect();
            cache.insert(v, format!("entry-{i}"), None).await;
        }
    });

    let query: Vec<f32> = (0..64).map(|j| (j as f32 * 0.1).cos()).collect();

    c.bench_function("vector_cache_search_500", |b| {
        b.iter(|| {
            rt.block_on(cache.search(black_box(&query), None));
        })
    });
}

// ── Groups ───────────────────────────────────────────────────────────

criterion_group!(
    exact_cache,
    bench_exact_cache_insert,
    bench_exact_cache_get_hit,
    bench_exact_cache_get_miss,
);

criterion_group!(
    vector_cache,
    bench_vector_cache_insert,
    bench_vector_cache_search,
);

criterion_main!(exact_cache, vector_cache);

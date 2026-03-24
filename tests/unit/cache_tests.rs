// =============================================================================
// tests/unit/cache_tests.rs — Unit-level tests for cache subsystem.
//
// Covers ExactMatchCache and VectorCache in isolation (no HTTP, no middleware).
// =============================================================================

use std::num::NonZeroUsize;

use isartor::layer1::layer1a_cache::ExactMatchCache;
use isartor::vector_cache::VectorCache;

// ═══════════════════════════════════════════════════════════════════════
// ExactMatchCache
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn exact_cache_put_and_get() {
    let cache = ExactMatchCache::new(NonZeroUsize::new(10).unwrap());
    cache.put("key1".into(), "value1".into());
    assert_eq!(cache.get("key1"), Some("value1".to_string()));
}

#[test]
fn exact_cache_miss_returns_none() {
    let cache = ExactMatchCache::new(NonZeroUsize::new(10).unwrap());
    assert_eq!(cache.get("nonexistent"), None);
}

#[test]
fn exact_cache_overwrite_existing_key() {
    let cache = ExactMatchCache::new(NonZeroUsize::new(10).unwrap());
    cache.put("key1".into(), "old".into());
    cache.put("key1".into(), "new".into());
    assert_eq!(cache.get("key1"), Some("new".to_string()));
}

#[test]
fn exact_cache_capacity_eviction() {
    // Capacity of 2 — inserting a 3rd should evict the LRU entry.
    let cache = ExactMatchCache::new(NonZeroUsize::new(2).unwrap());
    cache.put("a".into(), "1".into());
    cache.put("b".into(), "2".into());
    cache.put("c".into(), "3".into());

    // "a" should have been evicted (LRU).
    assert_eq!(cache.get("a"), None);
    assert_eq!(cache.get("b"), Some("2".to_string()));
    assert_eq!(cache.get("c"), Some("3".to_string()));
}

#[test]
fn exact_cache_get_promotes_entry() {
    // Access order affects LRU eviction.
    let cache = ExactMatchCache::new(NonZeroUsize::new(2).unwrap());
    cache.put("a".into(), "1".into());
    cache.put("b".into(), "2".into());

    // Access "a" so it becomes most-recently-used.
    let _ = cache.get("a");

    // Insert "c" — should evict "b" (the LRU), not "a".
    cache.put("c".into(), "3".into());
    assert_eq!(cache.get("a"), Some("1".to_string()));
    assert_eq!(cache.get("b"), None);
    assert_eq!(cache.get("c"), Some("3".to_string()));
}

#[test]
fn exact_cache_sha256_key_pattern() {
    // The cache middleware hashes prompts via SHA-256. Verify
    // that hex-encoded SHA-256 keys work correctly.
    use sha2::{Digest, Sha256};
    let cache = ExactMatchCache::new(NonZeroUsize::new(100).unwrap());
    let key = hex::encode(Sha256::digest(b"test prompt"));
    cache.put(key.clone(), r#"{"layer":1,"message":"cached"}"#.into());
    assert_eq!(
        cache.get(&key),
        Some(r#"{"layer":1,"message":"cached"}"#.to_string())
    );
}

// ═══════════════════════════════════════════════════════════════════════
// VectorCache — Semantic cache
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn vector_cache_insert_and_search() {
    let cache = VectorCache::new(0.8, 300, 100);
    let embedding = vec![1.0_f32, 0.0, 0.0];
    cache
        .insert(embedding.clone(), "semantic result".into(), None)
        .await;

    let result = cache.search(&embedding, None).await;
    assert_eq!(result, Some("semantic result".to_string()));
}

#[tokio::test]
async fn vector_cache_search_miss() {
    let cache = VectorCache::new(0.95, 300, 100);
    let v1 = vec![1.0_f32, 0.0, 0.0];
    let v2 = vec![0.0_f32, 1.0, 0.0]; // orthogonal → cosine ~0

    cache.insert(v1, "stored".into(), None).await;
    let result = cache.search(&v2, None).await;
    assert_eq!(result, None);
}

#[tokio::test]
async fn vector_cache_similar_vectors_hit() {
    let cache = VectorCache::new(0.9, 300, 100);
    let v1 = vec![1.0_f32, 0.0, 0.0];
    let similar = vec![0.99_f32, 0.1, 0.0]; // cosine ≈ 0.995

    cache.insert(v1, "semantic hit".into(), None).await;
    let result = cache.search(&similar, None).await;
    assert_eq!(result, Some("semantic hit".to_string()));
}

#[tokio::test]
async fn vector_cache_capacity_eviction() {
    // Capacity of 2.
    let cache = VectorCache::new(0.8, 300, 2);
    cache
        .insert(vec![1.0, 0.0, 0.0], "first".into(), None)
        .await;
    cache
        .insert(vec![0.0, 1.0, 0.0], "second".into(), None)
        .await;
    cache
        .insert(vec![0.0, 0.0, 1.0], "third".into(), None)
        .await;

    // The first entry should have been evicted.
    let r1 = cache.search(&[1.0, 0.0, 0.0], None).await;
    assert_eq!(r1, None, "first entry should be evicted");
    let r3 = cache.search(&[0.0, 0.0, 1.0], None).await;
    assert_eq!(r3, Some("third".to_string()));
}

#[tokio::test]
async fn vector_cache_empty_search() {
    let cache = VectorCache::new(0.8, 300, 100);
    let result = cache.search(&[1.0, 0.0, 0.0], None).await;
    assert_eq!(result, None);
}

#[tokio::test]
async fn vector_cache_ttl_expiry() {
    // TTL of 1 second.
    let cache = VectorCache::new(0.8, 1, 100);
    cache
        .insert(vec![1.0, 0.0, 0.0], "ephemeral".into(), None)
        .await;

    // Should be present immediately.
    assert_eq!(
        cache.search(&[1.0, 0.0, 0.0], None).await,
        Some("ephemeral".to_string())
    );

    // Wait for TTL to expire.
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    assert_eq!(
        cache.search(&[1.0, 0.0, 0.0], None).await,
        None,
        "entry should have expired"
    );
}

#[tokio::test]
async fn vector_cache_session_scope_prevents_cross_session_hits() {
    let cache = VectorCache::new(0.8, 300, 100);
    let embedding = vec![1.0_f32, 0.0, 0.0];
    cache
        .insert(
            embedding.clone(),
            "session result".into(),
            Some("session-a".to_string()),
        )
        .await;

    assert_eq!(
        cache.search(&embedding, Some("session-a")).await,
        Some("session result".to_string())
    );
    assert_eq!(cache.search(&embedding, Some("session-b")).await, None);
    assert_eq!(cache.search(&embedding, None).await, None);
}

use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// A single cached entry: the embedding vector paired with its response.
struct CacheEntry {
    embedding: Vec<f32>,
    response: String,
    created_at: Instant,
    session_cache_scope: Option<String>,
}

/// In-memory vector cache that uses cosine similarity to find semantically
/// similar prompts.
///
/// Uses brute-force cosine similarity scan which is perfectly adequate for
/// cache sizes typical in gateway workloads (< 10K entries, < 1ms per scan).
///
/// Stored behind a `RwLock` so that reads can proceed concurrently while
/// writes (inserts / evictions) hold exclusive access.
pub struct VectorCache {
    entries: RwLock<Vec<CacheEntry>>,
    similarity_threshold: f64,
    ttl: Duration,
    max_capacity: usize,
}

impl VectorCache {
    pub fn new(similarity_threshold: f64, ttl_secs: u64, max_capacity: u64) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            similarity_threshold,
            ttl: Duration::from_secs(ttl_secs),
            max_capacity: max_capacity as usize,
        }
    }

    /// Search for a cached response whose embedding is within the
    /// similarity threshold. Returns the best match if one exists.
    pub async fn search(&self, query: &[f32], session_cache_scope: Option<&str>) -> Option<String> {
        let span = tracing::info_span!(
            "l1b_semantic_cache_search",
            cache.entries_scanned = tracing::field::Empty,
            cache.hit = false,
            cosine_similarity = tracing::field::Empty,
        );

        // Acquire the read lock before entering the span so the non-Send
        // `SpanGuard` never lives across an `.await` point.
        let entries = self.entries.read().await;
        let _guard = span.enter();
        let now = Instant::now();

        let mut best: Option<(&CacheEntry, f64)> = None;
        let mut scanned: u64 = 0;

        for entry in entries.iter() {
            // Skip expired entries.
            if now.duration_since(entry.created_at) > self.ttl {
                continue;
            }
            if entry.session_cache_scope.as_deref() != session_cache_scope {
                continue;
            }
            scanned += 1;
            let score = cosine_similarity(query, &entry.embedding);
            if score >= self.similarity_threshold && best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((entry, score));
            }
        }

        span.record("cache.entries_scanned", scanned);

        if let Some((entry, score)) = best {
            span.record("cache.hit", true);
            span.record("cosine_similarity", format!("{score:.4}").as_str());
            tracing::info!(cosine_similarity = score, "Semantic cache hit");
            Some(entry.response.clone())
        } else {
            tracing::debug!("Semantic cache miss");
            None
        }
    }

    /// Insert a new embedding + response pair into the cache.
    /// Evicts expired entries and enforces the capacity cap.
    pub async fn insert(
        &self,
        embedding: Vec<f32>,
        response: String,
        session_cache_scope: Option<String>,
    ) {
        let span = tracing::debug_span!(
            "l1b_semantic_cache_insert",
            cache.evicted = tracing::field::Empty,
            cache.size_after = tracing::field::Empty,
        );

        // Acquire the write lock before entering the span.
        let mut entries = self.entries.write().await;
        let _guard = span.enter();
        let now = Instant::now();

        let before = entries.len();
        // Evict expired entries.
        entries.retain(|e| now.duration_since(e.created_at) <= self.ttl);
        let evicted = before - entries.len();

        // If still at capacity, remove the oldest entry.
        if entries.len() >= self.max_capacity {
            entries.remove(0);
        }

        entries.push(CacheEntry {
            embedding,
            response,
            created_at: now,
            session_cache_scope,
        });

        let size = entries.len();
        span.record("cache.evicted", evicted as u64);
        span.record("cache.size_after", size as u64);
        tracing::debug!(
            cache.size = size,
            evicted,
            "Semantic cache: inserted new entry"
        );
    }

    /// Return the current number of (non-expired) entries in the cache.
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Return true if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in \[-1, 1\] where 1 means identical direction.
/// Returns 0.0 if either vector has zero magnitude (degenerate case).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot: f64 = 0.0;
    let mut mag_a: f64 = 0.0;
    let mut mag_b: f64 = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }

    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── cosine_similarity tests ──────────────────────────────────

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-9);
    }

    #[test]
    fn cosine_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let score = cosine_similarity(&a, &b);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn cosine_mismatched_lengths() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let score = cosine_similarity(&a, &b);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&b, &a), 0.0);
    }

    #[test]
    fn cosine_scaled_vectors_same_score() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![2.0f32, 4.0, 6.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 1.0).abs() < 1e-9);
    }

    // ── VectorCache tests ────────────────────────────────────────

    #[tokio::test]
    async fn vector_cache_insert_and_search_hit() {
        let cache = VectorCache::new(0.9, 300, 100);
        let embedding = vec![1.0f32, 0.0, 0.0];
        cache
            .insert(embedding.clone(), "found it".into(), None)
            .await;
        let result = cache.search(&embedding, None).await;
        assert_eq!(result, Some("found it".into()));
    }

    #[tokio::test]
    async fn vector_cache_search_miss_below_threshold() {
        let cache = VectorCache::new(0.99, 300, 100);
        cache.insert(vec![1.0, 0.0, 0.0], "resp".into(), None).await;
        // Orthogonal vector — similarity ~0.0
        let query = vec![0.0f32, 1.0, 0.0];
        let result = cache.search(&query, None).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn vector_cache_search_empty_cache() {
        let cache = VectorCache::new(0.5, 300, 100);
        let query = vec![1.0f32, 0.0];
        let result = cache.search(&query, None).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn vector_cache_ttl_expiry() {
        let cache = VectorCache::new(0.5, 0, 100); // TTL = 0
        cache.insert(vec![1.0, 0.0], "cached".into(), None).await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let query = vec![1.0f32, 0.0];
        let result = cache.search(&query, None).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn vector_cache_capacity_eviction() {
        let cache = VectorCache::new(0.5, 300, 2);
        cache.insert(vec![1.0, 0.0], "first".into(), None).await;
        cache.insert(vec![0.0, 1.0], "second".into(), None).await;
        cache.insert(vec![0.5, 0.5], "third".into(), None).await;

        // First entry (index 0) should have been evicted.
        // The exact search for [1.0, 0.0] may still partially match
        // remaining entries but "first" should be gone.
        let entries = cache.entries.read().await;
        assert!(entries.len() <= 2);
        // "first" should not be in the cache anymore.
        assert!(!entries.iter().any(|e| e.response == "first"));
    }

    #[tokio::test]
    async fn vector_cache_returns_best_match() {
        let cache = VectorCache::new(0.5, 300, 100);
        // Insert two similar vectors.
        cache
            .insert(vec![1.0, 0.0, 0.0], "less similar".into(), None)
            .await;
        cache
            .insert(vec![1.0, 0.1, 0.0], "more similar".into(), None)
            .await;

        // Query is [1.0, 0.1, 0.0] — exact match for "more similar".
        let query = vec![1.0f32, 0.1, 0.0];
        let result = cache.search(&query, None).await;
        assert_eq!(result, Some("more similar".into()));
    }

    #[tokio::test]
    async fn vector_cache_isolated_by_session_scope() {
        let cache = VectorCache::new(0.5, 300, 100);
        let query = vec![1.0f32, 0.0];
        cache
            .insert(query.clone(), "scoped".into(), Some("scope-a".to_string()))
            .await;

        assert_eq!(
            cache.search(&query, Some("scope-a")).await,
            Some("scoped".into())
        );
        assert_eq!(cache.search(&query, Some("scope-b")).await, None);
        assert_eq!(cache.search(&query, None).await, None);
    }
}

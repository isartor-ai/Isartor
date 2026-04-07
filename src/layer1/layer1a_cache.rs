use ahash::RandomState;
use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Ultra-fast, concurrent, bounded exact match cache for Layer 1a.
pub struct ExactMatchCache {
    inner: Arc<RwLock<LruCache<String, String, RandomState>>>,
}

impl ExactMatchCache {
    /// Create a new cache with the given capacity (number of entries).
    pub fn new(capacity: NonZeroUsize) -> Self {
        let cache = LruCache::with_hasher(capacity, RandomState::new());
        Self {
            inner: Arc::new(RwLock::new(cache)),
        }
    }

    /// Get a cached response for the given prompt, promoting it to most-recent.
    pub fn get(&self, prompt: &str) -> Option<String> {
        // LruCache::get requires &mut self to promote, so we must briefly take a write lock.
        let mut cache = self.inner.write();
        cache.get(prompt).cloned()
    }

    /// Insert a prompt/response pair into the cache.
    pub fn put(&self, prompt: String, response: String) {
        let mut cache = self.inner.write();
        cache.put(prompt, response);
    }

    /// Return the current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Return true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn test_exact_match_cache_eviction() {
        let cache = ExactMatchCache::new(NonZeroUsize::new(2).unwrap());

        cache.put("Prompt A".to_string(), "Response A".to_string());
        cache.put("Prompt B".to_string(), "Response B".to_string());

        // Access A to make it most recently used.
        assert_eq!(cache.get("Prompt A"), Some("Response A".to_string()));

        // Insert C, should evict B (A was just accessed).
        cache.put("Prompt C".to_string(), "Response C".to_string());

        assert_eq!(cache.get("Prompt A"), Some("Response A".to_string()));
        assert_eq!(cache.get("Prompt B"), None);
        assert_eq!(cache.get("Prompt C"), Some("Response C".to_string()));
    }
}

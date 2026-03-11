//! Integration test for VectorCache HNSW ANN search
use isartor::vector_cache::VectorCache;

#[tokio::test]
async fn hnsw_ann_basic_search() {
    let cache = VectorCache::new(0.7, 300, 10);
    let v1 = vec![1.0f32, 0.0, 0.0];
    let v2 = vec![0.0f32, 1.0, 0.0];
    let v3 = vec![0.9f32, 0.1, 0.0];
    cache.insert(v1.clone(), "first".into()).await;
    cache.insert(v2.clone(), "second".into()).await;
    cache.insert(v3.clone(), "third".into()).await;

    // Query close to v3
    let query = vec![0.95f32, 0.05, 0.0];
    let result = cache.search(&query).await;
    assert_eq!(result, Some("third".into()));

    // Query close to v2
    let query2 = vec![0.0f32, 1.0, 0.0];
    let result2 = cache.search(&query2).await;
    assert_eq!(result2, Some("second".into()));
}

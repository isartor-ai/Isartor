// =============================================================================
// Stub Implementations — Placeholder algorithms for pipeline compilation.
//
// Each stub satisfies its trait contract with minimal logic. Replace these
// one-by-one with production implementations backed by real ML models.
// =============================================================================

use async_trait::async_trait;

use super::context::IntentClassification;
use super::traits::{
    Embedder, ExternalLlm, IntentClassifier, LocalExecutor, Reranker, VectorStore,
};

// ═════════════════════════════════════════════════════════════════════
// Layer 1 Stubs
// ═════════════════════════════════════════════════════════════════════

/// Stub embedder that produces a fixed-dimension zero vector.
///
/// Replace with: candle BertModel sentence-embedding implementation.
pub struct StubEmbedder {
    pub dimension: usize,
}

impl StubEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl Embedder for StubEmbedder {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f64>> {
        // Stub: produces a deterministic hash-based vector for testing.
        // Production code uses TextEmbedder (candle) or LlamaCppEmbedder instead.
        //
        // This stub produces a deterministic hash-based vector for testing.
        tracing::debug!(
            text_len = text.len(),
            dim = self.dimension,
            "StubEmbedder: generating pseudo-embedding"
        );

        let mut vector = vec![0.0f64; self.dimension];
        for (i, byte) in text.bytes().enumerate() {
            vector[i % self.dimension] += byte as f64 / 255.0;
        }
        // Normalise to unit length.
        let magnitude: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for v in &mut vector {
                *v /= magnitude;
            }
        }
        Ok(vector)
    }

    fn embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "stub-embedder-v0"
    }
}

// ─────────────────────────────────────────────────────────────────────

/// Stub vector store backed by a simple in-memory Vec with brute-force search.
///
/// Replace with: production VectorCache or external vector DB (Qdrant, Weaviate).
pub struct StubVectorStore {
    entries: tokio::sync::RwLock<Vec<(Vec<f64>, String)>>,
}

impl Default for StubVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl StubVectorStore {
    pub fn new() -> Self {
        Self {
            entries: tokio::sync::RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorStore for StubVectorStore {
    async fn search(
        &self,
        query_vector: &[f64],
        threshold: f64,
    ) -> anyhow::Result<Option<(String, f64)>> {
        // Stub: brute-force cosine similarity scan for testing.
        // Production code uses VectorCache or InMemoryVectorStore instead.
        let entries = self.entries.read().await;

        let mut best: Option<(String, f64)> = None;
        for (embedding, response) in entries.iter() {
            let sim = cosine_similarity(query_vector, embedding);
            if sim >= threshold && best.as_ref().is_none_or(|(_, s)| sim > *s) {
                best = Some((response.clone(), sim));
            }
        }

        Ok(best)
    }

    async fn insert(&self, embedding: Vec<f64>, response: String) -> anyhow::Result<()> {
        let mut entries = self.entries.write().await;
        entries.push((embedding, response));
        tracing::debug!(size = entries.len(), "StubVectorStore: inserted entry");
        Ok(())
    }

    async fn len(&self) -> usize {
        self.entries.read().await.len()
    }
}

// ═════════════════════════════════════════════════════════════════════
// Layer 2 Stubs
// ═════════════════════════════════════════════════════════════════════

/// Stub intent classifier using simple keyword heuristics.
///
/// Replace with: local SLM performing Zero-Shot NLI classification.
pub struct StubIntentClassifier;

#[async_trait]
impl IntentClassifier for StubIntentClassifier {
    async fn classify(&self, text: &str) -> anyhow::Result<(IntentClassification, f64)> {
        // Stub: naive keyword matching for testing.
        // Production implementations: EmbeddedClassifier (candle Zero-Shot NLI),
        //   LlamaCppClassifier (HTTP proxy to llama.cpp server).
        let lower = text.to_lowercase();

        let (intent, confidence) = if lower.contains("hello")
            || lower.contains("hi")
            || lower.contains("thanks")
            || lower.contains("what time")
            || lower.contains("who are you")
        {
            (IntentClassification::Simple, 0.92)
        } else if lower.contains("code")
            || lower.contains("implement")
            || lower.contains("function")
            || lower.contains("debug")
        {
            (IntentClassification::CodeGen, 0.85)
        } else if lower.contains("document")
            || lower.contains("search")
            || lower.contains("find")
            || lower.contains("knowledge")
        {
            (IntentClassification::Rag, 0.80)
        } else if lower.len() > 200 || lower.contains("explain") || lower.contains("analyze") {
            (IntentClassification::Complex, 0.75)
        } else {
            (IntentClassification::Complex, 0.60)
        };

        tracing::debug!(
            intent = %intent,
            confidence = confidence,
            "StubIntentClassifier: classified"
        );

        Ok((intent, confidence))
    }

    fn model_name(&self) -> &str {
        "stub-classifier-v0"
    }
}

// ─────────────────────────────────────────────────────────────────────

/// Stub local executor that returns a canned response.
///
/// Replace with: Ollama / local candle text generation model.
pub struct StubLocalExecutor;

#[async_trait]
impl LocalExecutor for StubLocalExecutor {
    async fn execute_simple(&self, prompt: &str) -> anyhow::Result<String> {
        // Stub: returns a synthetic response for testing.
        // Production implementation: LlamaCppExecutor (HTTP proxy to llama.cpp server).
        tracing::debug!(
            prompt_len = prompt.len(),
            "StubLocalExecutor: generating simple response"
        );
        Ok(format!(
            "[StubLocalExecutor] Simple response for: \"{}\"",
            truncate(prompt, 80),
        ))
    }

    fn model_name(&self) -> &str {
        "stub-executor-v0"
    }
}

// ═════════════════════════════════════════════════════════════════════
// Layer 2.5 Stubs
// ═════════════════════════════════════════════════════════════════════

/// Stub reranker that selects the first N documents (no actual scoring).
///
/// Replace with: Cross-Encoder model (e.g., ms-marco-MiniLM) via candle.
pub struct StubReranker;

#[async_trait]
impl Reranker for StubReranker {
    async fn rerank(
        &self,
        prompt: &str,
        documents: &[String],
        top_k: usize,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        // Stub: assigns a synthetic score based on keyword overlap.
        // Production implementation: LlamaCppReranker (HTTP proxy to llama.cpp
        //   serving a Cross-Encoder model like ms-marco-MiniLM).
        let prompt_words: std::collections::HashSet<&str> = prompt.split_whitespace().collect();

        let mut scored: Vec<(String, f64)> = documents
            .iter()
            .map(|doc| {
                let doc_words: std::collections::HashSet<&str> = doc.split_whitespace().collect();
                let overlap = prompt_words.intersection(&doc_words).count() as f64;
                let score = overlap / (prompt_words.len().max(1) as f64);
                (doc.clone(), score)
            })
            .collect();

        // Sort by descending relevance score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        tracing::debug!(
            input_docs = documents.len(),
            output_docs = scored.len(),
            "StubReranker: reranked documents"
        );

        Ok(scored)
    }

    fn model_name(&self) -> &str {
        "stub-reranker-v0"
    }
}

// ═════════════════════════════════════════════════════════════════════
// Layer 3 Stubs
// ═════════════════════════════════════════════════════════════════════

/// Stub external LLM that returns a synthetic response.
///
/// Replace with: integration to OpenAI / Anthropic / Azure via `rig-core`.
pub struct StubExternalLlm;

#[async_trait]
impl ExternalLlm for StubExternalLlm {
    async fn complete(&self, prompt: &str, context_documents: &[String]) -> anyhow::Result<String> {
        // Stub: echoes the prompt and context document count for testing.
        // Production implementations: LlamaCppExternalLlm (HTTP proxy),
        //   RigAgent (rig-core Agent with OpenAI / Anthropic / Azure providers).
        tracing::debug!(
            prompt_len = prompt.len(),
            context_docs = context_documents.len(),
            "StubExternalLlm: generating completion"
        );

        let context_summary = if context_documents.is_empty() {
            "no additional context".to_string()
        } else {
            format!("{} context documents included", context_documents.len())
        };

        Ok(format!(
            "[StubExternalLlm] Completion for: \"{}\" ({})",
            truncate(prompt, 80),
            context_summary,
        ))
    }

    fn provider_name(&self) -> &str {
        "stub"
    }

    fn model_name(&self) -> &str {
        "stub-llm-v0"
    }
}

// ═════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

/// Truncate a string to at most `max_len` characters, appending "…" if needed.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::IntentClassification;
    use crate::pipeline::traits::*;

    // ── StubEmbedder ─────────────────────────────────────────────

    #[tokio::test]
    async fn stub_embedder_produces_correct_dimension() {
        let embedder = StubEmbedder::new(128);
        let vector = embedder.embed("hello world").await.unwrap();
        assert_eq!(vector.len(), 128);
        assert_eq!(embedder.embedding_dimension(), 128);
    }

    #[tokio::test]
    async fn stub_embedder_normalized() {
        let embedder = StubEmbedder::new(64);
        let vector = embedder.embed("test").await.unwrap();
        let magnitude: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 1e-6,
            "Embedding should be L2-normalised"
        );
    }

    #[tokio::test]
    async fn stub_embedder_deterministic() {
        let embedder = StubEmbedder::new(32);
        let v1 = embedder.embed("same input").await.unwrap();
        let v2 = embedder.embed("same input").await.unwrap();
        assert_eq!(v1, v2, "Same input should produce same embedding");
    }

    #[tokio::test]
    async fn stub_embedder_different_inputs_different_outputs() {
        let embedder = StubEmbedder::new(32);
        let v1 = embedder.embed("hello").await.unwrap();
        let v2 = embedder.embed("world").await.unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn stub_embedder_model_name() {
        let embedder = StubEmbedder::new(128);
        assert_eq!(embedder.model_name(), "stub-embedder-v0");
    }

    // ── StubVectorStore ──────────────────────────────────────────

    #[tokio::test]
    async fn stub_vector_store_insert_and_search() {
        let store = StubVectorStore::new();
        let vec = vec![1.0, 0.0, 0.0];
        store.insert(vec.clone(), "found".into()).await.unwrap();

        let result = store.search(&vec, 0.8).await.unwrap();
        assert!(result.is_some());
        let (resp, score) = result.unwrap();
        assert_eq!(resp, "found");
        assert!(score >= 0.8);
    }

    #[tokio::test]
    async fn stub_vector_store_search_miss() {
        let store = StubVectorStore::new();
        store.insert(vec![1.0, 0.0, 0.0], "a".into()).await.unwrap();
        // Orthogonal vector → ~0.0 similarity.
        let query = vec![0.0f64, 1.0, 0.0];
        let result = store.search(&query, 0.99).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn stub_vector_store_len() {
        let store = StubVectorStore::new();
        assert_eq!(store.len().await, 0);
        assert!(store.is_empty().await);

        store.insert(vec![1.0], "a".into()).await.unwrap();
        assert_eq!(store.len().await, 1);
        assert!(!store.is_empty().await);
    }

    // ── StubIntentClassifier ─────────────────────────────────────

    #[tokio::test]
    async fn stub_classifier_simple() {
        let classifier = StubIntentClassifier;
        let (intent, confidence) = classifier.classify("hello").await.unwrap();
        assert_eq!(intent, IntentClassification::Simple);
        assert!(confidence > 0.0);
    }

    #[tokio::test]
    async fn stub_classifier_code() {
        let classifier = StubIntentClassifier;
        let (intent, _) = classifier.classify("write code in Rust").await.unwrap();
        assert_eq!(intent, IntentClassification::CodeGen);
    }

    #[tokio::test]
    async fn stub_classifier_complex() {
        let classifier = StubIntentClassifier;
        let (intent, _) = classifier.classify("explain quantum computing in depth and analyze the implications for modern cryptography").await.unwrap();
        assert_eq!(intent, IntentClassification::Complex);
    }

    #[test]
    fn stub_classifier_model_name() {
        let classifier = StubIntentClassifier;
        assert_eq!(classifier.model_name(), "stub-classifier-v0");
    }

    // ── StubLocalExecutor ────────────────────────────────────────

    #[tokio::test]
    async fn stub_local_executor_returns_response() {
        let executor = StubLocalExecutor;
        let result = executor.execute_simple("What is 2+2?").await.unwrap();
        assert!(result.contains("StubLocalExecutor"));
        assert!(result.contains("What is 2+2?"));
    }

    #[test]
    fn stub_local_executor_model_name() {
        let executor = StubLocalExecutor;
        assert_eq!(executor.model_name(), "stub-executor-v0");
    }

    // ── StubReranker ─────────────────────────────────────────────

    #[tokio::test]
    async fn stub_reranker_returns_top_k() {
        let reranker = StubReranker;
        let docs = vec![
            "Rust is a programming language".to_string(),
            "Cats are cute animals".to_string(),
            "Rust programming is systems level".to_string(),
        ];
        let result = reranker.rerank("Rust programming", &docs, 2).await.unwrap();
        assert_eq!(result.len(), 2);
        // Scores should be in descending order.
        assert!(result[0].1 >= result[1].1);
    }

    #[tokio::test]
    async fn stub_reranker_empty_docs() {
        let reranker = StubReranker;
        let docs: Vec<String> = vec![];
        let result = reranker.rerank("test", &docs, 5).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn stub_reranker_model_name() {
        let reranker = StubReranker;
        assert_eq!(reranker.model_name(), "stub-reranker-v0");
    }

    // ── StubExternalLlm ──────────────────────────────────────────

    #[tokio::test]
    async fn stub_external_llm_no_context() {
        let llm = StubExternalLlm;
        let docs: Vec<String> = vec![];
        let result = llm.complete("hello", &docs).await.unwrap();
        assert!(result.contains("StubExternalLlm"));
        assert!(result.contains("no additional context"));
    }

    #[tokio::test]
    async fn stub_external_llm_with_context() {
        let llm = StubExternalLlm;
        let docs = vec!["doc1".to_string(), "doc2".to_string()];
        let result = llm.complete("hello", &docs).await.unwrap();
        assert!(result.contains("2 context documents"));
    }

    #[test]
    fn stub_external_llm_names() {
        let llm = StubExternalLlm;
        assert_eq!(llm.provider_name(), "stub");
        assert_eq!(llm.model_name(), "stub-llm-v0");
    }

    // ── cosine_similarity helper ─────────────────────────────────

    #[test]
    fn stub_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stub_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn stub_cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn stub_cosine_mismatch_len() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    // ── truncate helper ──────────────────────────────────────────

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let result = truncate("hello world", 5);
        assert_eq!(result, "hello…");
    }

    #[test]
    fn truncate_exact_length() {
        assert_eq!(truncate("abcde", 5), "abcde");
    }
}

#![allow(dead_code)]
//! # Core Ports — Trait Interfaces for the Pluggable Trait Provider Pattern
//!
//! These traits define the **Ports** (in Hexagonal Architecture parlance) that
//! decouple Isartor's pipeline logic from any concrete implementation.
//!
//! Each trait has exactly two production adapters:
//!
//! | Port            | Minimalist (single binary)     | Enterprise (K8s)            |
//! |-----------------|--------------------------------|-----------------------------|
//! | `ExactCache`    | `InMemoryCache` (ahash + LRU)  | `RedisExactCache` (Redis)   |
//! | `SlmRouter`     | `EmbeddedCandleRouter` (candle)| `RemoteVllmRouter` (vLLM)   |
//!
//! The active adapter is chosen at startup via configuration (see `factory.rs`).

use async_trait::async_trait;

// ═══════════════════════════════════════════════════════════════════════
// Port: ExactCache — Layer 1a prompt→response exact-match cache
// ═══════════════════════════════════════════════════════════════════════

/// Asynchronous, thread-safe exact-match cache for Layer 1a.
///
/// Implementations may be purely in-memory (LRU) or backed by a distributed
/// store (Redis, Memcached, etc.). All methods are `Send + Sync` safe so
/// the cache can be shared across Tokio tasks via `Arc<dyn ExactCache>`.
#[async_trait]
pub trait ExactCache: Send + Sync {
    /// Look up a cached response by the SHA-256 hex key of the prompt.
    ///
    /// Returns `Ok(Some(response))` on a cache hit, `Ok(None)` on a miss,
    /// or an error if the backing store is unreachable.
    async fn get(&self, key: &str) -> anyhow::Result<Option<String>>;

    /// Store a prompt key → response pair in the cache.
    ///
    /// Implementations should handle eviction (LRU, TTL, etc.) internally.
    async fn put(&self, key: &str, response: &str) -> anyhow::Result<()>;
}

// ═══════════════════════════════════════════════════════════════════════
// Port: SlmRouter — Layer 2 intent classification / SLM triage
// ═══════════════════════════════════════════════════════════════════════

/// Asynchronous, thread-safe intent classifier for Layer 2 triage.
///
/// Implementations may run inference in-process (Candle / ONNX) or delegate
/// to a remote model serving endpoint (vLLM, TGI, llama.cpp sidecar).
#[async_trait]
pub trait SlmRouter: Send + Sync {
    /// Classify the user prompt into an intent label.
    ///
    /// Expected labels: `"SIMPLE"`, `"COMPLEX"`, `"RAG"`, `"CODEGEN"`.
    ///
    /// Implementations should return a normalised uppercase label string.
    /// On error the caller will fall through to the next pipeline layer.
    async fn classify_intent(&self, prompt: &str) -> anyhow::Result<String>;
}

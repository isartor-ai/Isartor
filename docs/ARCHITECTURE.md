# Isartor — Pluggable Trait Provider Architecture

> **Pattern:** Hexagonal Architecture (Ports & Adapters)
> **Location:** `src/core/`, `src/adapters/`, `src/factory.rs`

---

## Why Pluggable Adapters?

Isartor targets two radically different deployment profiles with the *same*
binary:

| Dimension | **Minimalist** (Edge / VPS) | **Enterprise** (K8s / Cloud) |
|---|---|---|
| Cache (L1a) | In-process LRU (ahash + parking_lot) | Redis cluster |
| Semantic (L1b) | Candle BertModel (in-process) | External TEI (optional) |
| Router (L2) | Embedded Candle GGUF inference | Remote vLLM / TGI server |
| Context Optimiser (L2.5) | In-process rerank (retrieve + rerank, e.g., top-K selection) | Distributed rerank (optional, e.g., TEI/ANN pool) |
| Deployment | Single static binary, `docker run` | Helm chart, horizontal auto-scaling |
| Dependencies | Zero external services | Redis, vLLM pods, Prometheus, Jaeger |

Rather than feature-flag every call-site, we define **Ports** (trait
interfaces) and swap the concrete **Adapter** at startup via configuration.
This keeps the Deflection Stack logic (middleware, handler, orchestrator) completely
agnostic to the backing implementation.

---

## Scalability Model

The pluggable adapter architecture is the foundation of Isartor's
scalability story. Each deployment tier builds on the previous one with
zero code changes:

```text
Level 1 (Edge)           Level 2 (Compose)        Level 3 (K8s)
┌────────────────┐       ┌────────────────┐       ┌────────────────┐
│ Single Process  │       │ Firewall + GPU  │       │ N Firewall Pods │
│ memory cache    │──▶    │ Sidecar         │──▶    │ + Redis Cluster │
│ embedded candle │       │ memory cache    │       │ + vLLM Pool     │
│ context optimiser │     │ (optional)      │       │ (optional)      │

---

## Layer 2.5 — Context Optimiser

Layer 2.5 is responsible for retrieving and reranking candidate documents or responses to minimize downstream token usage. This layer typically implements top-K selection, reranking, or context window optimization before forwarding to the LLM. It is instrumented as the `context_optimise` span in observability.
└────────────────┘       └────────────────┘       └────────────────┘
```

### Horizontal Scaling Considerations

| Component | Scaling Strategy | Bottleneck |
|---|---|---|
| **Firewall pods** | Stateless — scale horizontally via HPA on CPU / RPS | Each pod has its own in-memory LRU unless `cache_backend=redis` |
| **Exact cache** | `memory` → per-pod; `redis` → shared across all pods | Redis: network round-trip (~0.5 ms); Memory: no shared state |
| **Router (L2)** | `embedded` → per-pod CPU; `vllm` → shared GPU pool | Embedded: `Mutex` serialises inference per pod; vLLM: continuous batching |
| **Semantic cache** | In-process candle BertModel — per-pod, no sharing | Embedding model (~90 MB) loaded per pod |
| **Layer 3 (LLM)** | Cloud API — inherently scalable | Rate limits from the LLM provider |

**Key insight:** Switching to `cache_backend=redis` is what unlocks true
multi-replica scaling. Without it, each firewall pod maintains an independent
cache, leading to duplicated work and lower hit rates. With Redis, all pods
share the same cache keyspace.

---

## Directory Layout

```text
src/
├── core/
│   ├── mod.rs            # Re-exports
│   └── ports.rs          # Trait interfaces (ExactCache, SlmRouter)
├── adapters/
│   ├── mod.rs            # Re-exports
│   ├── cache.rs          # InMemoryCache, RedisExactCache
│   └── router.rs         # EmbeddedCandleRouter, RemoteVllmRouter
├── factory.rs            # build_exact_cache(), build_slm_router()
└── config.rs             # CacheBackend, RouterBackend enums + AppConfig
```

---

## Ports (Trait Interfaces)

Defined in `src/core/ports.rs`. Each port is an `async_trait` that requires
`Send + Sync` so it can be shared via `Arc<dyn Port>` across Tokio tasks.

### `ExactCache` — Layer 1a Exact-Match Cache

```rust
#[async_trait]
pub trait ExactCache: Send + Sync {
    async fn get(&self, key: &str) -> anyhow::Result<Option<String>>;
    async fn put(&self, key: &str, response: &str) -> anyhow::Result<()>;
}
```

### `SlmRouter` — Layer 2 Intent Classification

```rust
#[async_trait]
pub trait SlmRouter: Send + Sync {
    async fn classify_intent(&self, prompt: &str) -> anyhow::Result<String>;
}
```

---

## Adapters (Concrete Implementations)

### Cache Adapters (`src/adapters/cache.rs`)

| Adapter          | Backend                  | Use Case                          |
|------------------|--------------------------|-----------------------------------|
| `InMemoryCache`  | ahash + LRU + parking_lot| Single-binary edge, dev, tests    |
| `RedisExactCache`| Redis (skeleton)         | Multi-replica K8s, shared state   |

### Router Adapters (`src/adapters/router.rs`)

| Adapter               | Backend              | Use Case                          |
|-----------------------|----------------------|-----------------------------------|
| `EmbeddedCandleRouter`| Candle GGUF (in-proc)| Edge, VPS, offline inference      |
| `RemoteVllmRouter`    | vLLM HTTP endpoint   | GPU cluster, high-throughput      |

---

## Factory — Configuration-Driven Wiring

`src/factory.rs` is the **single place** where the port → adapter decision
is made. It reads `AppConfig` and returns `Arc<dyn Port>`:

```text
┌──────────────┐
│  AppConfig    │
│ cache_backend │──► Memory   → InMemoryCache
│               │──► Redis    → RedisExactCache
│router_backend │──► Embedded → EmbeddedCandleRouter
│               │──► Vllm     → RemoteVllmRouter
└──────────────┘
```

### API

```rust
pub fn build_exact_cache(config: &AppConfig) -> Arc<dyn ExactCache>;
pub fn build_slm_router(config: &AppConfig, http_client: &Client) -> Arc<dyn SlmRouter>;
```

---

## Configuration

The backend selection is controlled by two `AppConfig` fields that default
to the Minimalist profile:

| Env Variable                 | Config Field       | Default      | Values             |
|------------------------------|--------------------|--------------|--------------------|
| `ISARTOR__CACHE_BACKEND`     | `cache_backend`    | `memory`     | `memory`, `redis`  |
| `ISARTOR__ROUTER_BACKEND`    | `router_backend`   | `embedded`   | `embedded`, `vllm` |
| `ISARTOR__REDIS_URL`         | `redis_url`        | `redis://127.0.0.1:6379` | Any Redis URI |
| `ISARTOR__VLLM_URL`          | `vllm_url`         | `http://127.0.0.1:8000`  | vLLM base URL |
| `ISARTOR__VLLM_MODEL`        | `vllm_model`       | `gemma-2-2b-it` | Model name string  |

### Example: Enterprise Profile

```bash
export ISARTOR__CACHE_BACKEND=redis
export ISARTOR__REDIS_URL=redis://redis-cluster.svc:6379
export ISARTOR__ROUTER_BACKEND=vllm
export ISARTOR__VLLM_URL=http://vllm.svc:8000
export ISARTOR__VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
```

No code changes required — the factory instantiates the correct adapters at
startup based on these environment variables.

---

## Adding a New Adapter

1. **Define the struct** in `src/adapters/cache.rs` or `src/adapters/router.rs`.
2. **Implement the port trait** (`ExactCache` or `SlmRouter`).
3. **Add a variant** to the config enum (`CacheBackend` or `RouterBackend`)
   in `src/config.rs`.
4. **Wire it** in `src/factory.rs` with a new `match` arm.
5. **Write tests** — each adapter module has a `#[cfg(test)] mod tests`.

No other files need to change. The middleware and pipeline code operate only
on `Arc<dyn ExactCache>` / `Arc<dyn SlmRouter>`.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `Arc<dyn Trait>` over generics | Allows runtime backend selection from config; avoids monomorphisation bloat for adapters that are never used in a given deployment. |
| `async_trait` | Even synchronous adapters (InMemoryCache) implement async signatures so the pipeline doesn't need to know whether I/O is involved. |
| `anyhow::Result` | Adapters may fail (network, Redis timeout); the pipeline uses `?` to fall through gracefully. |
| Skeleton Redis / vLLM adapters | Ship the interface now, fill in the wire protocol later. Tests prove the trait contract is satisfied. |
| Factory in its own module | Keeps the `AppState` constructor clean and makes it trivial to add feature-flagged adapters later. |

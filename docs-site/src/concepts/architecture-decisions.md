# Architecture Decision Records

> **Key design decisions, trade-offs, and rationale behind Isartor's architecture.**

Each ADR follows a lightweight format: Context → Decision → Consequences.

Maintenance policy: when an implementation changes Isartor's lasting system structure, request flow, API surfaces, routing strategy, cache behavior, deployment model, or other architectural trade-offs, update this ADR page in the same change. If that implementation also changes user-visible capabilities, keep the feature list in `README.md`, supplementary docs in `docs/`, and published docs in `docs-site/src/` aligned as part of the same ticket.

---

## ADR-001: Multi-Layer Deflection Stack Architecture

**Date:** 2024 · **Status:** Accepted

### Context

AI Prompt Firewall traffic follows a power-law distribution: the majority of prompts are simple or repetitive, while only a small fraction requires expensive cloud LLMs. Sending all traffic to a single provider wastes tokens and money.

### Decision

Implement a **sequential Deflection Stack** with 4+ layers, each capable of short-circuiting:

- **Layer 0** — Operational defense (auth, rate limiting, concurrency control)
- **Layer 1** — Semantic + exact cache (zero-cost hits)
- **Layer 2** — Local SLM triage (classify intent, execute simple tasks locally)
- **Layer 2.5** — Context optimiser (instruction dedup + minification to reduce cloud input tokens)
- **Layer 3** — Cloud LLM fallback with ordered provider chain, quotas, and multi-key rotation

**Layer 2.5 (Context Optimiser):**
Compresses repeated instruction payloads (CLAUDE.md, copilot-instructions.md, skills blocks) via a modular `CompressionPipeline` with three built-in stages: ContentClassifier (gate), DedupStage (session-aware cross-turn dedup), and LogCrunchStage (static minification). Instrumented as the `layer2_5_context_optimizer` span in observability.

### Consequences

- **Positive:** 60–80% of traffic can be resolved before Layer 3, dramatically reducing cost.
- **Positive:** Each layer adds latency only when needed — cache hits are sub-millisecond.
- **Positive:** Clear separation of concerns; each layer is independently testable.
- **Negative:** Deflection Stack adds conceptual complexity vs. a simple reverse proxy.
- **Negative:** Each layer needs its own error handling and timeout strategy.

---

## ADR-002: Axum + Tokio as Runtime Foundation

**Date:** 2024 · **Status:** Accepted

### Context

The firewall must handle high concurrency (thousands of simultaneous connections) with low latency overhead. The binary should be small, statically linked, and deployable to minimal environments.

### Decision

Use **Axum 0.8** on **Tokio 1.x** for the async HTTP server. Build with `--target x86_64-unknown-linux-musl` and `opt-level = "z"` + LTO for a ~5 MB static binary.

### Consequences

- **Positive:** Tokio's work-stealing scheduler handles 10K+ concurrent connections efficiently.
- **Positive:** Axum's type-safe extractors catch errors at compile time.
- **Positive:** Static musl binary runs in distroless containers (no libc, no shell).
- **Negative:** Rust's compilation times are longer than Go/Node.js equivalents.
- **Negative:** Ecosystem is smaller — fewer off-the-shelf middleware components.

---

## ADR-003: Embedded Candle Classifier (Layer 2)

**Date:** 2024 · **Status:** Accepted

### Context

For minimal deployments (edge, VPS, air-gapped), requiring an external sidecar (llama.cpp, Ollama, TGI) adds operational complexity. Many classification tasks can be handled by a 2B parameter model on CPU.

### Decision

Embed a **Gemma-2-2B-IT** GGUF model directly in the Rust process using the [candle](https://github.com/huggingface/candle) framework. The model is loaded on first start via `hf-hub` (auto-downloaded from Hugging Face) and wrapped in a `tokio::sync::Mutex` for thread-safe inference on `spawn_blocking`.

### Consequences

- **Positive:** Zero external dependencies for Layer 2 classification — a single binary handles everything.
- **Positive:** No HTTP overhead for classification calls; inference is an in-process function call.
- **Positive:** Works in air-gapped environments with pre-cached models.
- **Negative:** ~1.5 GB memory overhead for the Q4_K_M model weights.
- **Negative:** CPU inference is slower than GPU (50–200 ms classification, 200–2000 ms generation).
- **Negative:** `Mutex` serialises inference calls — throughput limited to one inference at a time.
- **Trade-off:** For higher throughput, upgrade to Level 2 (llama.cpp sidecar on GPU).

---

## ADR-004: Three Deployment Tiers

**Date:** 2024 · **Status:** Accepted

### Context

Isartor targets a wide range of deployments, from a developer's laptop to enterprise Kubernetes clusters. A single deployment model cannot serve all use cases optimally.

### Decision

Define three explicit deployment tiers that share the **same binary and configuration surface**:

| Tier | Strategy | Target |
|:-----|:---------|:-------|
| **Level 1** | Monolithic binary, embedded candle | VPS, edge, bare metal |
| **Level 2** | Firewall + llama.cpp sidecars | Docker Compose, single host + GPU |
| **Level 3** | Stateless pods + inference pools | Kubernetes, Helm, HPA |

The tier is selected purely by environment variables and infrastructure, not by code changes.

### Consequences

- **Positive:** A single codebase and binary serves all deployment scenarios.
- **Positive:** Users start at Level 1 and upgrade incrementally — no migrations.
- **Positive:** Clear documentation entry points for each tier.
- **Negative:** Some config variables are irrelevant at certain tiers (e.g., `ISARTOR__LAYER2__SIDECAR_URL` is unused at Level 1 with embedded candle).
- **Negative:** Testing all three tiers requires different infrastructure setups.

---

## ADR-005: llama.cpp as Sidecar (Level 2) Instead of Ollama

**Date:** 2024 · **Status:** Accepted

### Context

The original design used [Ollama](https://ollama.com/) (~1.5 GB image) as the local SLM engine. While Ollama has a convenient API and model management, it's heavyweight for a sidecar.

### Decision

Replace Ollama with **llama.cpp server** (`ghcr.io/ggml-org/llama.cpp:server`, ~30 MB) as the default sidecar in `docker-compose.sidecar.yml`. Two instances run side by side:

- **slm-generation** (port 8081) — Phi-3-mini for classification and generation
- **slm-embedding** (port 8082) — all-MiniLM-L6-v2 with `--embedding` flag

### Consequences

- **Positive:** 50× smaller container images (30 MB vs. 1.5 GB).
- **Positive:** Faster cold starts; no model pull step needed (uses `--hf-repo` auto-download).
- **Positive:** OpenAI-compatible API — firewall code doesn't need to change.
- **Negative:** Ollama's model management UX (pull, list, delete) is lost.
- **Negative:** Each model needs its own llama.cpp instance (no multi-model serving).
- **Migration:** Ollama-based Compose files (`docker-compose.yml`, `docker-compose.azure.yml`) are retained for backward compatibility.
- **Update (ADR-011):** The **slm-embedding** sidecar (port 8082) is now **optional**. Layer 1 semantic cache embeddings are generated in-process via candle (pure-Rust BertModel).

---

## ADR-006: rig-core for Multi-Provider LLM Client

**Date:** 2024 · **Status:** Accepted

### Context

Layer 3 must route to multiple cloud LLM providers (OpenAI, Azure OpenAI, Anthropic, xAI). Implementing each provider's API client from scratch would be error-prone and hard to maintain.

### Decision

Use [rig-core](https://crates.io/crates/rig-core) (v0.32.0) as the unified LLM client. Rig provides a consistent `CompletionModel` abstraction over all supported providers.

### Consequences

- **Positive:** Single configuration surface (`ISARTOR__LLM_PROVIDER` + `ISARTOR__EXTERNAL_LLM_API_KEY`) switches providers.
- **Positive:** Provider-specific quirks (Azure deployment IDs, Anthropic versioning) handled by rig.
- **Negative:** Adds a dependency; rig's release cadence may not match our needs.
- **Negative:** Limited to providers rig supports (but covers all major ones).

---

## ADR-007: AIMD Adaptive Concurrency Control

**Date:** 2024 · **Status:** Accepted

### Context

A fixed concurrency limit either over-provisions (wasting resources) or under-provisions (rejecting requests during traffic spikes). The firewall needs to dynamically adjust its limit based on real-time latency.

### Decision

Implement an **Additive Increase / Multiplicative Decrease (AIMD)** concurrency limiter at Layer 0:

- If P95 latency < target → `limit += 1` (additive increase).
- If P95 latency > target → `limit *= 0.5` (multiplicative decrease).
- Bounded by configurable min/max concurrency limits.

### Consequences

- **Positive:** Self-tuning: the limit converges to the optimal value for the current load.
- **Positive:** Protects downstream services (sidecars, cloud LLMs) from overload.
- **Negative:** During cold start, the limit starts low and ramps up — initial requests may see 503s.
- **Tuning:** Target latency must be calibrated per deployment tier.

---

## ADR-008: Unified API Surface

**Date:** 2024 · **Status:** Superseded

### Context

The original design maintained two API versions: a v1 middleware-based pipeline (`/api/chat`) and a v2 orchestrator-based pipeline (`/api/v2/chat`). Maintaining two code paths increased complexity with no clear benefit once the middleware pipeline matured.

### Decision

Consolidate into a single endpoint:

- **`/api/chat`** — Middleware-based pipeline. Each layer is an Axum middleware (auth → cache → SLM triage → handler).
- The v2 endpoint (`/api/v2/chat`) and its `pipeline_*` configuration fields have been removed.
- Orchestrator and trait-based pipeline components remain in `src/pipeline/` for potential future reintegration.

### Consequences

- **Positive:** Single code path to maintain, test, and observe.
- **Positive:** Simplified configuration surface — no more `PIPELINE_*` env vars.
- **Positive:** Eliminates user confusion about which endpoint to use.
- **Negative:** Orchestrator-based features (structured `processing_log`, explicit `PipelineContext`) are not exposed until reintegrated.

---

## ADR-009: Distroless Container Image

**Date:** 2024 · **Status:** Accepted

### Context

The firewall binary is statically linked (musl). The runtime container only needs to execute a single binary.

### Decision

Use `gcr.io/distroless/static-debian12` as the runtime base image. It contains no shell, no package manager, no libc — only the static binary.

### Consequences

- **Positive:** Minimal attack surface — no shell to exec into, no tools for attackers.
- **Positive:** Tiny image size (base ~2 MB + binary ~5 MB = ~7 MB total).
- **Positive:** Passes most container security scanners with zero CVEs.
- **Negative:** Cannot `docker exec` into the container for debugging (no shell).
- **Negative:** Cannot install additional tools at runtime.
- **Workaround:** Use `docker logs`, Jaeger traces, and Prometheus metrics for debugging.

---

## ADR-010: OpenTelemetry for Observability

**Date:** 2024 · **Status:** Accepted

### Context

The firewall needs distributed tracing and metrics. Vendor-specific SDKs (Datadog, New Relic, etc.) create lock-in.

### Decision

Use **OpenTelemetry** (OTLP gRPC) as the sole telemetry interface. Traces and metrics are exported to an OTel Collector, which can forward to any backend (Jaeger, Prometheus, Grafana, Datadog, etc.).

### Consequences

- **Positive:** Vendor-neutral — switch backends by reconfiguring the collector, not the app.
- **Positive:** OTLP is a CNCF standard with wide ecosystem support.
- **Positive:** When `ISARTOR__ENABLE_MONITORING=false`, no OTel SDK is initialised — zero overhead.
- **Negative:** Requires an OTel Collector as middleware (adds one more service in Level 2/3).
- **Negative:** Auto-instrumentation is less mature in Rust than in Java/Python.

---

## ADR-011: Pure-Rust Candle for In-Process Sentence Embeddings

| | |
|:--|:--|
| **Status** | Accepted (superseded: fastembed → candle) |
| **Date** | 2025-06 (updated 2025-07) |
| **Deciders** | Core team |
| **Relates to** | ADR-003 (Embedded Candle), ADR-005 (llama.cpp sidecar) |

### Context

Layer 1 (semantic cache) must generate sentence embeddings for every incoming prompt to compute cosine similarity against the vector cache. Previously, this was done via **fastembed** (ONNX Runtime, BAAI/bge-small-en-v1.5), which introduced a C++ dependency (onnxruntime-sys) that broke cross-compilation on ARM64 macOS and complicated the build matrix.

### Decision

Use **candle** (`candle-core`, `candle-nn`, `candle-transformers` 0.9) with **hf-hub** and **tokenizers** to run **sentence-transformers/all-MiniLM-L6-v2** in-process via a pure-Rust `BertModel`. The model weights (~90 MB) are downloaded once from Hugging Face Hub on first startup and cached in `~/.cache/huggingface/`. Inference is invoked through `tokio::task::spawn_blocking` since BERT forward passes are CPU-bound.

- **Model:** sentence-transformers/all-MiniLM-L6-v2 — 384-dimensional embeddings, optimised for sentence similarity.
- **Runtime:** Pure-Rust candle stack — zero C/C++ dependencies, seamless cross-compilation to any `rustc` target.
- **Pooling:** Mean pooling with attention mask, followed by L2 normalisation.
- **Thread safety:** The inner `BertModel` is wrapped in `std::sync::Mutex` because `forward()` takes `&mut self`. This is acceptable because inference is always called from `spawn_blocking`, never holding the lock across `.await` points.
- **Architecture:** `TextEmbedder` is initialised once at startup, stored as `Arc<TextEmbedder>` in `AppState`, and injected into the cache middleware.

### Alternatives Considered

| Alternative | Why rejected |
|:------------|:-------------|
| fastembed (ONNX Runtime) | C++ dependency (onnxruntime-sys) breaks ARM64 cross-compilation; ~5 MB shared library |
| llama.cpp sidecar (all-MiniLM-L6-v2) | Network round-trip on hot path, extra container to manage |
| sentence-transformers (Python) | Crosses FFI boundary, adds Python runtime dependency |
| ort (raw ONNX Runtime bindings) | Same C++ dependency problem as fastembed |

### Consequences

- **Positive:** Eliminates ~2–5 ms network latency per embedding call on the cache hot path.
- **Positive:** Zero C/C++ dependencies — `cargo build` works on any platform without cmake or pre-built binaries.
- **Positive:** Zero sidecar dependency for Level 1 — the minimal Dockerfile runs self-contained.
- **Positive:** Model weights are auto-downloaded from Hugging Face Hub; reproducible builds.
- **Negative:** First startup downloads model weights (~90 MB) if not pre-cached.
- **Negative:** `Mutex` serialises concurrent embedding calls within a single process (acceptable at current scale; can be replaced with a pool of models if needed).

---

## ADR-012: Pluggable Trait Provider (Hexagonal Architecture)

| | |
|:--|:--|
| **Status** | Accepted |
| **Date** | 2025-06 |
| **Deciders** | Core team |
| **Relates to** | ADR-003 (Embedded Candle), ADR-004 (Three Deployment Tiers) |

### Context

As Isartor grew from a single-process binary (Level 1) to a multi-tier deployment (Level 1 → 2 → 3), the cache and SLM router components became tightly coupled to their in-process implementations. Scaling to Level 3 (Kubernetes, multiple replicas) requires:

1. **Shared cache** — in-process LRU caches are isolated per pod; cache hits are inconsistent, duplicating work.
2. **GPU-backed inference** — in-process Candle inference is CPU-bound; Level 3 needs a dedicated GPU inference pool (vLLM / TGI) that can scale independently.

Hard-coding these choices into the firewall binary would require compile-time feature flags or code branching, making the binary non-portable across tiers.

### Decision

Adopt the **Ports & Adapters (Hexagonal Architecture)** pattern:

- **Ports** (`src/core/ports.rs`) — Define `ExactCache` and `SlmRouter` as `async_trait` traits (`Send + Sync`), representing the interfaces the firewall depends on.
- **Adapters** (`src/adapters/`) — Provide concrete implementations:
  - `InMemoryCache` (ahash + LRU + parking_lot) and `RedisExactCache` for `ExactCache`
  - `EmbeddedCandleRouter` and `RemoteVllmRouter` for `SlmRouter`
- **Factory** (`src/factory.rs`) — `build_exact_cache(&config)` and `build_slm_router(&config, &http_client)` read `AppConfig.cache_backend` and `AppConfig.router_backend` at startup and return the appropriate `Box<dyn Trait>`.
- **Configuration** (`src/config.rs`) — `CacheBackend` enum (`Memory | Redis`) and `RouterBackend` enum (`Embedded | Vllm`) with associated connection URLs, selectable via `ISARTOR__CACHE_BACKEND` and `ISARTOR__ROUTER_BACKEND` env vars.

The **same binary** serves all three deployment tiers; the runtime behaviour is entirely configuration-driven.

### Alternatives Considered

| Alternative | Why rejected |
|:------------|:-------------|
| Compile-time feature flags (`#[cfg(feature = "redis")]`) | Produces different binaries per tier; complicates CI and container builds |
| Service mesh sidecar (Envoy filter for caching) | Adds infrastructure complexity; cache logic is domain-specific |
| Plugin system (dynamic `.so` loading) | Over-engineered; `dyn Trait` with compile-time-known variants is simpler |
| Runtime scripting (Lua / Wasm policy) | Unnecessary indirection; Rust trait dispatch is zero-cost |

### Consequences

- **Positive:** One binary, all tiers — only env vars change between Level 1 (embedded everything) and Level 3 (Redis + vLLM).
- **Positive:** Horizontal scalability — with `cache_backend=redis`, all pods share the same cache; with `router_backend=vllm`, GPU inference scales independently.
- **Positive:** Testability — unit tests inject mock adapters via the trait interface.
- **Positive:** Extensibility — adding a new backend (e.g., Memcached, Triton) requires only a new adapter implementing the trait.
- **Negative:** Minor runtime overhead from `dyn Trait` dynamic dispatch (single vtable lookup per call — negligible vs. network I/O).
- **Negative:** `EmbeddedCandleRouter` remains a skeleton; full candle-based classification requires the `embedded-inference` feature flag to be completed.

---

## ADR-013: Resolve Model Aliases Before Routing and Cache Key Generation

| | |
|:--|:--|
| **Status** | Accepted |
| **Date** | 2026-03 |
| **Deciders** | Core team |
| **Relates to** | ADR-001 (Deflection Stack), ADR-006 (rig-core), ADR-012 (Hexagonal Architecture) |

### Context

Users increasingly connect heterogeneous tools and SDKs to Isartor, but raw provider model IDs are verbose and change over time. Operators want stable names such as `fast`, `smart`, or `code` without fragmenting cache behavior or forcing every client to know the real provider model identifier.

### Decision

Add a `model_aliases` configuration map in `AppConfig` and resolve aliases at the HTTP boundary before:

- request routing to Layer 3
- exact-cache key generation
- semantic-cache input generation
- OpenAI-compatible `GET /v1/models` discovery output

Aliases currently resolve within the already-configured provider. They do not yet switch providers; multi-provider named routing remains a follow-on roadmap item.

### Consequences

- **Positive:** Clients can use stable, human-friendly model names without changing provider-specific IDs everywhere.
- **Positive:** Alias traffic and canonical model traffic share the same cache behavior because cache inputs are normalized first.
- **Positive:** `GET /v1/models` can advertise both real model IDs and operator-defined aliases.
- **Negative:** This introduces another routing indirection layer that operators must document clearly.
- **Negative:** Provider switching by alias is intentionally out of scope for this ADR and remains future work.

---

## ADR-014: Keep Request/Response Debug Logging Separate from Telemetry

| | |
|:--|:--|
| **Status** | Accepted |
| **Date** | 2026-03 |
| **Deciders** | Core team |
| **Relates to** | ADR-001 (Deflection Stack), ADR-010 (OpenTelemetry for Observability) |

### Context

Operators sometimes need to inspect the exact request and response payloads Isartor handled when debugging provider integrations, auth failures, or client compatibility issues. Existing OpenTelemetry spans record request metadata and routing outcomes, but intentionally do not include raw bodies because prompts often contain sensitive data and large payloads.

### Decision

Add a separate, opt-in request logging mode controlled by:

- `enable_request_logs = true|false`
- `request_log_path = "~/.isartor/request_logs"`

When enabled, the outer monitoring middleware writes one JSONL record per request/response exchange to rotating files under `request_log_path`. Sensitive headers such as `Authorization`, `api-key`, and `x-api-key` are redacted automatically, and body logging stays out of the normal `isartor.log` / OpenTelemetry stream. The CLI exposes these logs via `isartor logs --requests`.

### Consequences

- **Positive:** Troubleshooting provider and client integration issues becomes much faster because operators can inspect exact payloads.
- **Positive:** Normal operational logs and telemetry remain privacy-safer by default because body logging is opt-in and stored separately.
- **Positive:** The `logs` CLI can tail request debug logs without mixing them into startup/runtime logs.
- **Negative:** Even with redaction, request logs can contain sensitive prompt content, so access controls and retention need operator attention.
- **Negative:** Request logging adds some I/O overhead when enabled and should not be the default steady-state mode.

---

## ADR-015: Treat Additional OpenAI-Compatible Vendors as Registry Entries, Not Unique Protocols

| | |
|:--|:--|
| **Status** | Accepted |
| **Date** | 2026-03 |
| **Deciders** | Core team |
| **Relates to** | ADR-006 (rig-core for Multi-Provider LLM Client), ADR-013 (Resolve Model Aliases Before Routing and Cache Key Generation) |

### Context

More providers now expose OpenAI-compatible chat-completions APIs. Adding each of them as a bespoke protocol integration would create repetitive code across configuration, runtime routing, setup flows, and connectivity checks even when the wire format is effectively the same.

### Decision

Add new vendors such as Cerebras, Nebius, SiliconFlow, Fireworks, NVIDIA, and Chutes as named entries in Isartor's provider registry with:

- a stable `llm_provider` enum variant
- a default OpenAI-compatible endpoint
- default model suggestions for CLI/setup
- connectivity checks that probe each vendor's model-list endpoint
- runtime Layer 3 routing that reuses the OpenAI-compatible Rig client with a provider-specific base URL

### Consequences

- **Positive:** Users get first-class provider names in `set-key`, `setup`, and `check` without manually discovering endpoints.
- **Positive:** Runtime behavior stays simple because OpenAI-compatible providers share one implementation path where practical.
- **Positive:** Future provider additions become mostly registry work instead of bespoke transport work.
- **Negative:** Operators may assume every provider has identical semantics just because the protocol is OpenAI-compatible; model naming and auth policies still vary by vendor.

---

## ADR-016: Keep Provider Health Status In-Memory and Expose It Through Debug/CLI Views

| | |
|:--|:--|
| **Status** | Accepted |
| **Date** | 2026-03 |
| **Deciders** | Core team |
| **Relates to** | ADR-001 (Deflection Stack), ADR-010 (OpenTelemetry for Observability), ADR-015 (OpenAI-Compatible Provider Registry) |

### Context

Operators need a fast way to confirm which Layer 3 provider Isartor is using right now and whether recent upstream requests have been succeeding or failing. Existing `isartor check` output is a point-in-time connectivity probe, while prompt/agent stats focus on traffic history rather than the currently configured provider's health state.

### Decision

Track provider health in memory on `AppState` for the active provider only. The tracker records:

- configured provider name, model, and effective endpoint summary
- whether an API key / endpoint is configured
- request count and error count
- last-known success timestamp
- last-known error timestamp and compact error message

Expose that snapshot through two surfaces:

- authenticated `GET /debug/providers`
- `isartor providers` CLI output (with local-config fallback when the endpoint is unavailable)

The tracker is updated by real Layer 3 request outcomes and resets when the process restarts. It is not persisted to Redis, telemetry backends, or disk.

### Consequences

- **Positive:** Operators get a fast "what provider is active and is it healthy?" answer without tailing logs.
- **Positive:** The health view stays tightly aligned with the running process because it is updated directly from Layer 3 success/failure paths.
- **Positive:** Debug output remains lightweight and privacy-safer than request-body logging because it stores compact status metadata only.
- **Negative:** Health state is process-local, so multi-replica deployments must query each instance (or aggregate externally) if they want a fleet-wide view.
- **Negative:** Restarting Isartor clears the counters and timestamps by design.

---

## ADR-017: Use an Ordered Layer 3 Provider Chain with Per-Provider Retry Budgets

**Date:** 2026 · **Status:** Accepted

### Context

Operators want Isartor to remain deflection-first while becoming more resilient to upstream provider outages, rate limits, and quota exhaustion. The previous Layer 3 path retried inside a single provider only, so once that provider stayed unavailable the request failed even when a healthy backup provider/model pair was available.

### Decision

Keep the existing top-level `llm_provider` and `external_llm_*` settings as the primary Layer 3 backend, and add an ordered `fallback_providers` chain for optional backups. Each provider gets its own retry budget. When the current provider exhausts retries with a retry-safe upstream error (for example `429`, `5xx`, timeout, or quota-style failure), Isartor advances to the next configured provider. Bad-request style failures do not cascade.

Successful Layer 3 responses now include an `x-isartor-provider` response header naming the provider that actually answered. `isartor check` and the provider-health views also expose the full configured chain rather than only the primary provider.

### Consequences

- **Positive:** Isartor can keep serving complex prompts through a backup provider without changing Layer 1 / Layer 2 behavior.
- **Positive:** Operators can express resilient provider/model combos directly in config while preserving backward compatibility for single-provider setups.
- **Positive:** The cache remains provider-agnostic, so a prompt answered by a backup provider still populates the same exact-cache path for future hits.
- **Negative:** Layer 3 routing is more complex because retry policy and failover policy are now separate concerns.
- **Negative:** Fallback chains can hide provider drift if operators do not watch `x-isartor-provider` or the provider-status surfaces.

---

## ADR-018: Support Per-Provider Multi-Key Rotation with Cooldown

**Date:** 2026 · **Status:** Accepted

### Context

Some operators hold multiple credentials for the same upstream provider: personal keys, team-shared keys, or separate quota buckets. A single-key model makes Isartor brittle under rate limits because the request fails even when another valid key for the same provider/model is available.

### Decision

Allow each Layer 3 provider entry to define a `provider_keys` pool alongside the legacy single `external_llm_api_key` / `api_key` field. Keep single-key config backward compatible by treating the legacy key as an implicit pool member. Within a provider, Isartor now supports `round_robin` and `priority` selection plus a per-provider cooldown window after `429` or quota-style upstream failures.

Provider failover and key rotation are separate layers:

- retry within a provider can rotate to another key
- provider failover still happens only after that provider exhausts retries or hits a cascade-safe terminal error

Expose masked key-pool state through `isartor check` and the provider-status surfaces so operators can see which keys are configured and whether any are cooling down.

### Consequences

- **Positive:** Isartor can survive provider-side throttling without immediately changing providers or failing the request.
- **Positive:** Existing single-key configs continue to work unchanged.
- **Positive:** Operators gain visibility into key-pool strategy and cooldown state without logging raw secrets.
- **Negative:** Runtime state is more complex because provider health and key health are now separate but related views.
- **Negative:** Cooldown state remains process-local and resets on restart.

---

## ADR-019: Accept Gemini-Native Inbound API Traffic at the Gateway Boundary

**Date:** 2026 · **Status:** Accepted

### Context

Isartor already accepted native, OpenAI-compatible, and Anthropic-compatible client traffic, while Gemini support existed only as an upstream Layer 3 provider. That forced Gemini-native clients to rely on compatibility shims even when they naturally speak `generateContent` / `streamGenerateContent`.

### Decision

Add Gemini-native inbound routes at the HTTP boundary:

- `POST /v1beta/models/{model}:generateContent`
- `POST /v1beta/models/{model}:streamGenerateContent`

These routes reuse the same Layer 1 / Layer 2 / Layer 3 stack as the other surfaces, but keep their own `gemini` cache namespace so response shapes never cross-pollinate with native, OpenAI, or Anthropic cache entries. The model embedded in the request path becomes the canonical request model for alias resolution and cache-key generation when the body does not already provide one.

Successful non-streaming responses are returned in Gemini `GenerateContentResponse` JSON shape. Successful streaming responses are framed as Gemini-style SSE at the boundary while cached state remains canonical JSON.

### Consequences

- **Positive:** Gemini-native tools can point at Isartor directly without an OpenAI compatibility shim.
- **Positive:** Request caching and model-alias normalization stay consistent with the existing boundary design.
- **Positive:** Streaming and non-streaming Gemini traffic share one canonical cache representation, reducing duplicate logic.
- **Negative:** The gateway boundary now has another protocol surface to preserve and regression-test.
- **Negative:** Gemini tool/function semantics are not yet a separate passthrough path the way OpenAI tool calls are.

---

*← Back to [Architecture](architecture.md)*


## ADR-020: Persist provider/model usage analytics

- Status: Accepted
- Date: 2026-03-29

### Context

Operators need a lightweight way to understand actual L3 spend, deflection savings, and recent request volume by provider/model without adding an external billing pipeline.

### Decision

Persist append-only usage events to `usage.jsonl`, aggregate them in-process with retention pruning, and expose summaries through the authenticated debug API and `isartor stats --usage`. Deflected requests are recorded as saved cost against the configured primary provider/model, while actual cloud calls record estimated prompt/completion usage against the provider/model that served the request.

### Consequences

- Operators get simple local cost visibility with no extra services.
- Estimates remain heuristic when upstream providers do not return token counts.
- The usage log becomes part of the local operator surface and should be treated as operational data.

---

## ADR-021: Reuse the persisted usage tracker for provider quota enforcement

- Status: Accepted
- Date: 2026-03-29

### Context

Operators need quota guardrails per Layer 3 provider, including warning thresholds, hard blocks, and the ability to spill over to the next configured fallback provider. Creating a second persistence layer for quotas would risk drift between what Isartor reports as spend and what it enforces at request time.

### Decision

Use the persisted usage tracker as the source of truth for quota enforcement. Providers can define `[quota.<provider>]` policies with daily, weekly, and monthly token and/or cost limits, a `warning_threshold_ratio`, and an `action_on_limit` of `warn`, `block`, or `fallback`. Quota evaluation happens before Layer 3 dispatch using current-period usage plus projected request usage, and period windows reset on UTC boundaries.

### Consequences

- Usage reporting and quota enforcement stay aligned because both read the same event history.
- `fallback` quota actions integrate naturally with the existing ordered Layer 3 provider chain.
- Projected token/cost enforcement remains heuristic when upstream usage metadata is unavailable before the request is sent.

---

## ADR-022: Embedded web management dashboard

- Status: Accepted
- Date: 2026-04-08

### Context

Operators increasingly want a browser-based way to monitor the gateway — deflection rate, provider health, token costs, and the live request log — without needing to run additional services or install third-party tools. The CLI (`isartor stats`, `isartor providers`) covers this use-case well for power users, but a visual overview is friendlier for shared team environments and periodic spot-checks.

### Decision

Embed a single-page application (SPA) directly in the binary using Rust's `include_str!` macro. The HTML/CSS/JS file is self-contained (no CDN, no external assets) and served at `GET /dashboard/`. Admin API endpoints (`/api/admin/*`) are protected by the existing gateway API-key auth middleware and supply JSON to the frontend. The API key is stored in `sessionStorage` — it is never transmitted to any third party.

The dashboard provides five tabs:

| Tab | Key features |
|:--|:--|
| **Overview** | Deflection rate sparkline (7-day SVG chart), uptime pill, L1a/L1b cache entry counts, quota-warning banner, provider/model cards |
| **Providers** | Health per provider, per-key pool status, connectivity test (`POST /api/admin/providers/test`), Add Provider modal |
| **Usage** | Window summary cards, daily request bar chart, per-provider/model breakdown table, per-provider quota status |
| **Request Log** | Last 100 JSONL request-log entries, expandable rows showing full JSON details |
| **Configuration** | Form-based editor for all `isartor.toml` settings; `toml_edit` write preserves comments; restart-required banner on save |

### Consequences

- Zero runtime dependencies: no separate web server, no static-file mount, no CDN.
- The dashboard binary footprint is bounded by the size of the HTML/JS/CSS — currently around 40 KB (including logo PNG).
- Operators gain browser-level visibility and basic config management with the same API key already in use.
- Configuration writes go to `isartor.toml` on disk; a gateway restart is required to apply changes (hot-reload is not supported).
- `AppState` gains a `started_at: Instant` field for uptime reporting; all struct-literal test fixtures must include this field.
| **Request Log** | Last 100 JSONL request-log entries, expandable rows showing full JSON details |
| **Configuration** | Form-based editor for all `isartor.toml` settings; `toml_edit` write preserves comments; restart-required banner on save |

### Consequences

- Zero runtime dependencies: no separate web server, no static-file mount, no CDN.
- The dashboard binary footprint is bounded by the size of the HTML/JS/CSS — currently around 40 KB (including logo PNG).
- Operators gain browser-level visibility and basic config management with the same API key already in use.
- Configuration writes go to `isartor.toml` on disk; a gateway restart is required to apply changes (hot-reload is not supported).
- `AppState` gains a `started_at: Instant` field for uptime reporting; all struct-literal test fixtures must include this field.

## ADR-023: Bidirectional Format Translation Matrix

**Status:** Accepted  
**Date:** 2026-04-07

### Context

Isartor serves five distinct client wire formats:

| Client | Endpoint / Detection |
|--------|---------------------|
| Native (Isartor) | `POST /api/chat`, `POST /api/v1/chat` |
| OpenAI | `POST /v1/chat/completions` |
| Anthropic | `POST /v1/messages` |
| Gemini | `POST /v1beta/models/*:generateContent` |
| Cursor | `POST /v1/chat/completions` + `X-Cursor-Checksum` header |
| Kiro (AWS) | `POST /v1/chat/completions` + `X-Kiro-Version` header |

Previously, each client format was handled ad hoc inside its own handler function with no shared abstraction. Cross-format translation (e.g. an Anthropic client routed to a Groq provider) went through the Rig agent string extraction path, which loses structured tool-call information and multi-turn context.

### Decision

Introduce `src/formats/` as a ports-and-adapters layer for client wire formats:

- **`types.rs`** — canonical `InternalRequest`, `InternalResponse`, `InternalChunk`, `InternalMessage`, `InternalContent`, `InternalTool`  
- **`mod.rs`** — `ApiFormat` trait: `parse_request`, `build_response`, `cache_namespace`, `name`  
- **`openai.rs`** — reference implementation; also exports `internal_to_openai_body`  
- **`anthropic.rs`** — Anthropic Messages API; also exports `internal_to_anthropic_body`  
- **`gemini.rs`** — Gemini GenerateContent; also exports `internal_to_gemini_body`  
- **`cursor.rs`** — thin wrapper over OpenAI adapter; separate cache namespace  
- **`kiro.rs`** — thin wrapper over OpenAI adapter; separate cache namespace  
- **`translate.rs`** — `ProviderWireFormat` enum + `translate_request(req, provider)` → bytes

Format detection (`formats::detect_format`) checks path first, then headers. `formats::cache_namespace` (header-aware) replaces the path-only `cache_namespace_for_path` in the cache middleware.

### Cache namespace invariant

Each client format owns its cache namespace:

| Format | Namespace |
|--------|-----------|
| OpenAI | `openai` |
| Anthropic | `anthropic` |
| Gemini | `gemini` |
| Cursor | `cursor` |
| Kiro | `kiro` |
| Native | `native` |

Cursor and Kiro are now isolated from the generic OpenAI namespace so IDE-specific prompts do not collide with generic API traffic.

### SSE streaming

Handlers always return canonical JSON. The cache middleware (`src/middleware/cache.rs`) converts JSON → SSE at the boundary when `is_streaming` is true. `streaming_cache_response` is extended to handle `"cursor"` and `"kiro"` namespaces (both use OpenAI SSE format).

### Consequences

- `cache_namespace_for_path` is no longer used in `cache.rs`; it remains in `prompt.rs` for other callers but is superseded by `formats::cache_namespace` in the middleware.
- Adding a new client format requires: one new `src/formats/<name>.rs` file implementing `ApiFormat`, one entry in `formats::detect_format`, one entry in `formats::cache_namespace`, and one entry in `streaming_cache_response`.
- The Rig-based extract-text path remains as the fallback for all formats; the format module provides infrastructure for future passthrough-with-translation.

## ADR-024: Generalized Provider Authentication and Encrypted Local Credential Storage

**Status:** Accepted  
**Date:** 2026-04-07

### Context

Before this change, Isartor handled provider authentication in two disconnected ways:

- most providers expected a static `api_key` in `isartor.toml` or environment variables
- GitHub Copilot had bespoke device-flow logic embedded directly in `src/providers/copilot.rs`

That made interactive authentication inconsistent across providers and encouraged operators to keep long-lived credentials in config files even when an OAuth-style login flow existed.

### Decision

Introduce a shared `src/auth/` module with:

- **`OAuthProvider` trait** — common interface for device-flow login, token polling, refresh, and manual API-key capture
- **`TokenStore`** — AES-256-GCM encrypted credential files under `~/.isartor/tokens/`
- **`device_flow.rs`** — shared RFC 8628 polling loop and terminal instructions
- **provider implementations** for Copilot, Gemini, Kiro, Anthropic, and OpenAI
- **CLI entry point**: `isartor auth <provider>`, `isartor auth status`, `isartor auth logout <provider>`

Layer 3 provider resolution now does a best-effort lookup in the token store when a provider has no explicit configured `api_key`, so authenticated local credentials can be reused without copying them into `isartor.toml`.

### Consequences

- OAuth-capable providers now share one authentication framework instead of embedding login logic inside a single provider adapter.
- Stored credentials are kept outside the main config file and encrypted at rest on disk.
- Static `set-key` configuration remains supported for service accounts, CI, and headless deployments.
- Providers without a public device flow (currently Anthropic and OpenAI) still participate via the same encrypted store, but use secure terminal key entry instead of browser/device auth.

## ADR-025: Optional End-to-End Encrypted Cloud Config Sync

**Status:** Accepted  
**Date:** 2026-04-07

### Context

Operators who use Isartor across multiple machines needed a way to share provider/model configuration without manually copying `isartor.toml` and without uploading plaintext API keys to a hosted control plane.

The key constraints were:

- sync must be strictly opt-in
- the server must never see plaintext config
- self-hosting must be possible with the same binary
- OAuth tokens, cache contents, and usage history must remain local-only

### Decision

Introduce `src/sync/` plus the `isartor sync` CLI:

- **`isartor sync init`** — saves a local sync profile (`~/.isartor/sync-profile.json`)
- **`isartor sync push`** — filters the shareable subset of `isartor.toml`, encrypts it client-side with PBKDF2-derived AES-256-GCM, and uploads the encrypted blob
- **`isartor sync pull`** — downloads, decrypts, and merges only the syncable keys/tables back into the local config
- **`isartor sync serve`** — runs a self-hostable zero-knowledge blob server with `GET/PUT /sync/{user_hash}`

The sync server stores only `{ user_hash, salt_hex, encrypted_blob_hex, updated_at }`. Conflict detection is timestamp-based: push checks whether the remote record changed since the last local sync, and pull checks for concurrent local edits since the last pull. Manual override is explicit via `--force`.

### Consequences

- Config sync remains off by default; nothing leaves the machine unless the operator runs `isartor sync ...`.
- The synced subset is intentionally narrower than the full runtime config: provider settings, aliases, fallback chain, and quota/pricing preferences are included; OAuth tokens, cache contents, usage history, bind addresses, and local log/cache paths are excluded.
- The same Isartor binary can act as either a sync client or a self-hosted sync server, so no separate control-plane service is required for small deployments.

## ADR-026: Provider Health Includes Manual Tests and Optional Background Pings

**Status:** Accepted  
**Date:** 2026-04-08

### Context

The dashboard already showed provider health, but the signal only changed after real routed Layer 3 traffic succeeded or failed. That left two gaps:

- operators could click **Test** and confirm a provider was reachable, but the badge still stayed `unknown`
- quiet environments had stale health state for long periods because no request traffic exercised every configured provider

### Decision

Keep provider health in-memory, but allow two additional probe sources to update that same health snapshot:

- **manual dashboard tests** via `POST /api/admin/providers/test`
- **background periodic pings** driven by `provider_health_check_interval_secs` (default `300`, `0` disables)

Probe results update last success/failure timestamps and healthy/failing status without inflating routed request or error counters. The runtime spawns the periodic loop only when offline mode is disabled.

### Consequences

- Dashboard badges reflect successful manual tests immediately.
- Operators can keep provider status warm even during idle periods without sending full chat traffic through each provider.
- Health counters continue to represent real routed traffic, while probe events affect only liveness-oriented status fields.

## ADR-027: MiniLM Multi-Head Classifier Runs Before Cache and Layer 3 Routing

**Status:** Accepted  
**Date:** 2026-04-08

### Context

Issue #99 adds a second kind of local classification problem: not just "can Layer 2 answer this prompt locally?" but also "which provider/model should receive this request if it reaches Layer 3?" The codebase already loads `all-MiniLM-L6-v2` for L1b semantic cache, so introducing a second embedding model would add startup cost, memory pressure, and another source of drift.

Routing also cannot safely happen after Layer 1 cache lookup, because provider-directed routing changes which upstream answer is valid for a given request. Reusing the old cache key would let a classifier-routed request return a cached response created for a different provider path.

### Decision

Add an optional **Layer 0.5 MiniLM classifier-routing middleware** between auth and cache:

- reuse the existing in-process MiniLM embedder already loaded for L1b
- load a JSON artifact containing four lightweight linear heads: `task_type`, `complexity`, `persona`, and `domain`
- classify buffered request context via `extract_classifier_context()`
- match ordered config rules that can prefer a provider and/or override the request model before Layer 1 and Layer 3
- prefix exact/semantic cache material with the selected provider fragment when routing changes the provider
- support two operating modes:
  - **fallback mode** (`fallback_to_existing_routing = true`) — classifier failures or no-match results fall through to the old routing path
  - **fail-closed mode** (`fallback_to_existing_routing = false`) — classifier failures or no-match results return `503`

### Consequences

- The same MiniLM embedding now feeds both L1b semantic cache and classifier-guided Layer 3 routing, avoiding a second encoder.
- Classifier-selected provider/model choices become part of cache safety, so cache behavior is provider-aware when routing rules apply.
- Operators can roll the feature out gradually with fallback enabled, then tighten it to fail closed once artifacts and rules are trusted.
- A **model matrix** shorthand (`[classifier_routing.matrix]`) provides a visual 2D grid mapping `complexity × task_type` to `"provider/model"` targets, compiled into rules at startup. Explicit `rules` always take priority. The `"local"` target keeps a cell on the cache/SLM path. The `"default"` key in either dimension acts as a wildcard.

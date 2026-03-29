# The Deflection Stack

Every incoming request passes through a sequence of smart computing layers. Only prompts requiring genuine, complex reasoning survive the Deflection Stack to reach the cloud.

```text
Request ──► L1a Exact Cache ──► L1b Semantic Cache ──► L2 SLM Router ──► L2.5 Context Optimiser ──► L3 Cloud Logic
                 │ hit                │ hit                 │ simple             │ compressed                │
                 ▼                    ▼                     ▼                    ▼                           ▼
              Response             Response            Local Response     Optimised Prompt            Cloud Response
```

## Layers at a Glance

| Layer | Algorithm / Mechanism | What It Does | Typical Latency |
|:------|:----------------------|:-------------|:----------------|
| **L1a — Exact Cache** | Fast Hashing (`ahash`) | Sub-millisecond duplicate detection. Traps infinite agent loops instantly. | < 1 ms |
| **L1b — Semantic Cache** | Cosine Similarity (Embeddings) | Computes mathematical meaning via pure-Rust `candle` models (`all-MiniLM-L6-v2`) to catch variations ("Price?" ≈ "Cost?"). | 1–5 ms |
| **L2 — SLM Router** | Neural Classification (LLM) | Triages intent using an embedded Small Language Model (e.g. Qwen-1.5B) to resolve simple data extraction tasks. | 50–200 ms |
| **L2.5 — Context Optimiser** | Instruction Dedup + Minify | Compresses repeated instruction files (CLAUDE.md, copilot-instructions.md) via session dedup and static minification to reduce cloud input tokens. | < 1 ms |
| **L3 — Cloud Logic** | Provider Chain + Retries | Routes surviving complex prompts to 23+ providers (OpenAI, Anthropic, Azure, Gemini, etc.) with ordered fallback chain, per-provider retry budgets, multi-key rotation, and quota enforcement. | Network-bound |

Layers 1a and 1b deflect **71% of repetitive agentic traffic** (FAQ/agent loop patterns) and **38% of diverse task traffic** before any neural inference runs.

## Layer Details

### L1a — Exact Cache

**Algorithm:** Fast hashing with `ahash`

L1a is the first line of defence. It computes a hash of the incoming prompt and checks it against an in-memory LRU cache (single-binary mode) or a shared Redis cluster (enterprise mode).

- **Hit:** Returns the cached response immediately (sub-millisecond).
- **Miss:** The request continues to L1b.

Cache keys are namespaced before hashing (`native|prompt`, `openai|prompt`, `anthropic|prompt`, etc.) to ensure one endpoint never returns another endpoint's response schema. On a cache hit, `ChatResponse.layer` is normalised to `1` regardless of which layer originally produced the response.

| Mode | Implementation |
|:-----|:---------------|
| Minimalist | In-memory LRU (`ahash` + `parking_lot`) |
| Enterprise | Redis cluster (shared across replicas, async `redis` crate) |

### L1b — Semantic Cache

**Algorithm:** Cosine similarity over sentence embeddings (`all-MiniLM-L6-v2`)

L1b catches semantically equivalent prompts that differ in wording. A sentence embedding is computed for the incoming prompt using a pure-Rust `candle` BertModel, then compared against the vector cache using cosine similarity.

- **Hit (similarity above threshold):** Returns the cached response (1–5 ms).
- **Miss:** The request continues to L2.

**Embedding pipeline:**

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional embeddings (~90 MB).
- **Runtime:** Pure-Rust candle stack — zero C/C++ dependencies.
- **Pooling:** Mean pooling with attention mask, followed by L2 normalisation.
- **Thread safety:** `BertModel` is wrapped in `std::sync::Mutex`; inference runs on `tokio::task::spawn_blocking`.
- **Architecture:** `TextEmbedder` is initialised once at startup, stored as `Arc<TextEmbedder>` in `AppState`.

The vector cache is maintained in tandem with exact cache entries. Insertions and evictions update the index automatically, providing sub-millisecond vector search latency for thousands of embeddings.

| Mode | Implementation |
|:-----|:---------------|
| Minimalist | In-process `candle` BertModel |
| Enterprise | External TEI sidecar (optional) |

### L2 — SLM Router

**Algorithm:** Neural classification via Small Language Model

L2 runs a lightweight language model to classify the prompt's intent. Simple requests (data extraction, FAQ-style queries) can be resolved locally without reaching the cloud.

- **Simple intent:** Returns a locally generated response (50–200 ms).
- **Complex intent:** The request continues to L2.5.
- **Disabled (`enable_slm_router = false`):** Layer is a no-op; request falls through to L3.

| Mode | Implementation |
|:-----|:---------------|
| Minimalist | Embedded `candle` GGUF inference (e.g. Gemma-2-2B-IT, CPU) |
| Enterprise | Remote vLLM / TGI server (GPU pool) |

### L2.5 — Context Optimiser

**Algorithm:** CompressionPipeline — Modular staged compression

Agentic coding tools (Copilot, Claude Code, Cursor) send large instruction files
(CLAUDE.md, copilot-instructions.md, skills blocks) with every turn.  L2.5 detects
and compresses these payloads before they reach the cloud, saving input tokens on
every L3 call.

**Pipeline architecture (`src/compression/`):**

L2.5 uses a modular `CompressionPipeline` with pluggable stages that execute in
order.  Each stage is a stateless `CompressionStage` trait object.  If a stage sets
`short_circuit = true`, subsequent stages are skipped.

**Built-in stages (run in order):**

1. **ContentClassifier** — Gate stage: detects instruction vs conversational content.
   Short-circuits on conversational messages so downstream stages skip work.
2. **DedupStage** — Session-aware cross-turn deduplication.  Hashes instruction
   content per session; on repeat turns, replaces with a compact hash reference.
   Short-circuits on dedup hit.
3. **LogCrunchStage** — Static minification: strips HTML/XML comments, decorative
   horizontal rules, consecutive blank lines, and Unicode box-drawing decoration.

**Adding custom stages:**

Implement the `CompressionStage` trait and add your stage to the pipeline via
`build_pipeline()` in `src/compression/optimize.rs`.

**Configuration:**

| Variable | Default | Description |
|:---------|:--------|:------------|
| `ISARTOR__ENABLE_CONTEXT_OPTIMIZER` | `true` | Master switch for L2.5 |
| `ISARTOR__CONTEXT_OPTIMIZER_DEDUP` | `true` | Enable cross-turn instruction deduplication |
| `ISARTOR__CONTEXT_OPTIMIZER_MINIFY` | `true` | Enable static minification |

**Observability:**

- **Instrumented as:** `layer2_5_context_optimizer` span in distributed traces.
- **Response header:** `x-isartor-context-optimized: bytes_saved=<N>` on optimised requests.
- Span fields: `context.bytes_saved`, `context.strategy` (e.g. "classifier+dedup", "classifier+log_crunch").

| Mode | Implementation |
|:-----|:---------------|
| Minimalist | In-process CompressionPipeline (classifier → dedup → log_crunch) |
| Enterprise | In-process CompressionPipeline (extensible with custom stages) |

### L3 — Cloud Logic

**Algorithm:** Ordered provider chain with per-provider retry budgets

L3 is the final layer. Only the hardest prompts — those not resolved by cache, SLM, or context optimisation — reach the external cloud LLMs.

**Provider chain execution:**

1. Isartor evaluates **quota** for the current provider (daily/weekly/monthly token and cost windows). If the provider is over quota, the request either blocks (429), warns, or falls through to the next provider depending on the `action_on_limit` policy.
2. The request is dispatched to the provider with **retry logic** (exponential backoff, jitter). Each provider has its own independent retry budget.
3. On exhausting retries with a retry-safe upstream error (429, 5xx, timeout), Isartor advances to the **next fallback provider** in the chain.
4. Successful responses are annotated with the `x-isartor-provider` header.

**Multi-key rotation:** Each provider can own an in-memory key pool. When multiple credentials are configured, keys are selected with `round_robin` or `priority` strategy. Only the rate-limited key is cooled down after 429/quota failures — other keys continue serving.

**Supported providers (23+):**

- **Full client:** OpenAI, Azure OpenAI, Anthropic, Copilot (GitHub), Gemini, Cohere, xAI
- **OpenAI-compatible registry:** Groq, Cerebras, Nebius, SiliconFlow, Fireworks, NVIDIA, Chutes, DeepSeek, Galadriel, Hyperbolic, HuggingFace, Mira, Moonshot, Ollama, OpenRouter, Perplexity, Together

**Safety nets:**

- **Offline mode (`offline_mode = true`):** Blocks L3 routing explicitly with HTTP 503.
- **Stale fallback:** On L3 failure, checks the namespaced exact-cache key first, then a legacy un-namespaced key for backward compatibility.

| Mode | Implementation |
|:-----|:---------------|
| Minimalist | Direct to cloud providers via `rig-core` |
| Enterprise | Direct to cloud providers via `rig-core` |

## How Layers Interact

The deflection stack is implemented as Axum middleware plus a final handler. For authenticated routes, the execution order is:

1. **Body buffer** — `BufferedBody` stores the request body so multiple layers can read it.
2. **Request-level monitoring** — Observability instrumentation.
3. **Auth** — API key validation.
4. **Layer 1 cache** — L1a exact match, then L1b semantic match.
5. **Layer 2 SLM triage** — Intent classification and local response.
6. **Layer 2.5 context optimiser** — Instruction dedup + minification via `CompressionPipeline`.
7. **Layer 3 handler** — Cloud LLM fallback.

> **Implementation note:** Axum middleware wraps inside-out — the last `.layer(...)` added runs first. The stack order in `src/main.rs` documents this explicitly and must be preserved.

Public health routes (`/health`, `/healthz`) intentionally bypass the deflection stack. The authenticated routes are `/api/chat`, `/api/v1/chat`, `/v1/chat/completions`, `/v1/messages`, and `/v1beta/models/{model}:generateContent`.

## See Also

- [Architecture](architecture.md) — high-level system design and pluggable providers
- [Architecture Decision Records](architecture-decisions.md) — rationale behind the deflection stack design (ADR-001)
- [Configuration Reference](../configuration/reference.md)

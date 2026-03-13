# Layer 1a: Exact-match Cache (Updated)

Layer 1a now uses a high-performance, concurrent, bounded LRU cache (`ExactMatchCache`) implemented with `ahash`, `lru`, and `parking_lot`. This replaces the legacy async `ExactCache` (HashMap-based).

**Current Implementation:**

```text
ExactMatchCache (sync, LRU, ahash, parking_lot)
```

All Layer 1a cache operations are synchronous and thread-safe. The cache is part of `AppState` and is accessed directly in the Layer 1a middleware. All legacy `ExactCache` code and references have been removed.

## Migration Notes

- Removed: `ExactCache` (async, HashMap-based)
- Added: `ExactMatchCache` (sync, LRU, ahash, parking_lot)
- Updated: `AppState`, `middleware/cache.rs`, and all usages to use the new cache API.

## Pluggable Trait Provider Pattern

Isartor uses a Hexagonal Architecture (Ports & Adapters) for cache and
router backends. Trait interfaces ("Ports") in `src/core/ports.rs` decouple
the pipeline from any concrete implementation. A factory in `src/factory.rs`
reads `AppConfig` and wires the correct adapter at startup.

| Port | Minimalist Adapter | Enterprise Adapter |
|------|--------------------|--------------------|
| `ExactCache` | `InMemoryCache` (ahash + LRU) | `RedisExactCache` (Redis) |
| `SlmRouter` | `EmbeddedCandleRouter` (Candle) | `RemoteVllmRouter` (vLLM) |

Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
# Isartor — Prompt Firewall Architecture

## High-Level Overview

```mermaid
flowchart TB
    subgraph Clients
        C1["Mobile App"]
        C2["Web App"]
        C3["CLI / Agent"]
    end

    subgraph GW["Isartor Prompt Firewall (Single Binary)"]
        direction TB

        L0["Layer 0 — Authentication\n<code>middleware/auth.rs</code>\nValidates X-API-Key header"]
        L1["Layer 1 — Semantic Cache\n<code>middleware/cache.rs</code>\nEmbed → cosine similarity → cache hit"]
        L2["Layer 2 — SLM Triage\n<code>middleware/slm_triage.rs</code>\nIntent classification (embedded or sidecar)"]
        L25["Layer 2.5 — Context Optimiser\nRetrieve docs → Rerank to top-K"]
        L3["Layer 3 — Fallback Handler\n<code>handler.rs</code>\nRoutes to external LLM"]

        L0 --> L1 --> L2 --> L25 --> L3
    end

    subgraph Backends
        FE["candle BertModel\nall-MiniLM-L6-v2\nIn-process sentence embeddings"]
        EC["Embedded Classifier\n(candle · Gemma-2-2B-IT GGUF)\nIn-process, zero network hop"]
        SLM["llama.cpp Sidecar\n(external HTTP, optional)"]
        LLM["External LLM API\n(OpenAI · Azure · Anthropic · xAI)"]
        VS["Vector Store\n(in-memory HNSW)"]
    end

    C1 & C2 & C3 -->|"HTTPS"| L0
    L0 -.->|"401 Unauthorized"| C1 & C2 & C3
    L1 -.->|"Cache Hit"| C1 & C2 & C3
    L1 ---|"Embed (in-process)"| FE
    L1 ---|"Lookup"| VS
    L2 -->|"In-process"| EC
    L2 -.->|"HTTP (fallback)"| SLM
    L2 -.->|"Simple Task\nShort-Circuit"| C1 & C2 & C3
    L3 -->|"POST /chat"| LLM
    L3 -.->|"LLM Response"| C1 & C2 & C3
```

## The Deflection Stack

Each layer can **short-circuit** the Deflection Stack by returning a response directly, avoiding unnecessary downstream work.

```mermaid
sequenceDiagram
    participant Client
    participant L0 as Layer 0<br/>Auth
    participant L1 as Layer 1<br/>Cache
    participant FE as candle BertModel<br/>(in-process)
    participant L2 as Layer 2<br/>SLM Triage
    participant L3 as Layer 3<br/>LLM Fallback
    participant EC as Embedded Classifier<br/>(candle)
    participant SLM as llama.cpp Sidecar<br/>(optional)
    participant LLM as External LLM

    Client->>L0: POST /api/chat + X-API-Key
    alt Invalid / missing key
        L0-->>Client: 401 Unauthorized ⛔
    else Valid key
        L0->>L1: forward request
        L1->>FE: embed(prompt) via spawn_blocking
        FE-->>L1: 384-dim vector
        alt Cache hit
            L1-->>Client: Cached response ⚡
        else Cache miss
            L1->>L2: forward request
            alt Embedded classifier enabled
                L2->>EC: classify(prompt)
                EC-->>L2: label + confidence
            else Sidecar mode
                L2->>SLM: POST /v1/chat/completions
                SLM-->>L2: simple / complex
            end
            alt Simple task
                L2->>EC: execute(prompt)
                EC-->>L2: short response
                L2-->>Client: Local response 🟢
            else Complex task
                L2->>L3: forward request
                L3->>LLM: Generate completion
                LLM-->>L3: LLM response
                L3-->>Client: LLM response 🔵
            end
        end
    end
```

## Module Map

```mermaid
graph LR
    subgraph src
        main["main.rs\nBootstrap & Router"]
        config["config.rs\nAppConfig"]
        handler["handler.rs\nLayer 3 Handler"]
        subgraph layer1
            emb_l1["embeddings.rs\nTextEmbedder\n(candle BertModel)"]
        end
        subgraph middleware
            mod["mod.rs"]
            auth["auth.rs\nLayer 0"]
            cache["cache.rs\nLayer 1"]
            slm["slm_triage.rs\nLayer 2"]
        end
        subgraph services
            li["local_inference.rs\nEmbeddedClassifier\n(candle + Gemma-2)"]
        end
        subgraph pipeline
            ctx["context.rs\nIntentClassification"]
            traits["traits.rs\nEmbedder, Classifier, …"]
            orch["orchestrator.rs\nPipeline engine"]
            conc["concurrency.rs\nAdaptive limiter"]
            subgraph implementations
                emb["embedder.rs"]
                vs["vector_store.rs"]
                ic["intent_classifier.rs"]
                le["local_executor.rs"]
                rr["reranker.rs"]
                el["external_llm.rs"]
            end
        end
    end

    main --> config
    main --> handler
    main --> mod
    main --> li
    main --> emb_l1
    mod --> auth & cache & slm
    cache -.-> emb_l1
    auth -.-> config
    slm -.-> config
    orch --> traits
    orch --> ctx
    traits -.-> ic & le & emb & vs & rr & el
    li -.->|"implements"| traits
```

## Embedded Classifier — In-Process ML Inference

The **Embedded Classifier** (`src/services/local_inference.rs`) is a Rust-native ML inference engine that replaces the external llama.cpp sidecar for Layer 2 (intent classification) and simple-task execution. It loads a quantised Gemma-2-2B-IT model directly into the process using the [candle](https://github.com/huggingface/candle) framework.

### Why Embedded?

| Aspect | llama.cpp Sidecar | Embedded Classifier |
|---|---|---|
| **Deployment** | Extra container + HTTP port | Zero additional processes |
| **Latency** | Network round-trip (~1–5 ms overhead) | Direct function call (~0 ms overhead) |
| **Failure modes** | Sidecar crash, connection refused, timeouts | Only OOM or model corruption |
| **Observability** | External process logs | Rust-native `tracing` spans with token-level metrics |
| **Memory** | Separate process memory | Shared process heap (~1.5 GB for Q4_K_M) |

### Architecture

```
EmbeddedClassifier
├── tokenizer: Arc<Tokenizer>          ← HuggingFace tokenizers (tokenizer.json)
├── model: Arc<Mutex<ModelWeights>>    ← candle quantised GGUF weights
├── device: Device::Cpu                ← CPU inference (VPS-compatible)
├── config: EmbeddedClassifierConfig   ← repo_id, gguf_filename, max_tokens, …
└── eot_token_id: u32                  ← Gemma-2 <end_of_turn> token
```

### Inference Flow

```
User Prompt
    │
    ▼
┌──────────────────────────────────┐
│ Format with Gemma-2 chat template│
│ <start_of_turn>user\n            │
│ {system_prompt}\n{user_query}    │
│ <end_of_turn>\n                  │
│ <start_of_turn>model\n           │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│ Tokenise (HF Tokenizers)         │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│ Prefill: forward(prompt_tokens)  │  ← Single batched forward pass
│ Decode loop:                     │  ← One token at a time
│   forward([token], index_pos)    │
│   sample_token(logits)           │  ← Greedy argmax + rep. penalty
│   break on <end_of_turn> or max  │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│ Detokenise → raw text            │
│ Parse LABEL: / CONFIDENCE:       │  ← (classification only)
└──────────────────────────────────┘
```

### Thread Safety

`ModelWeights::forward()` requires `&mut self` to update internal KV-cache masks. Since pipeline traits use `&self`, the model is wrapped in `tokio::sync::Mutex`. All CPU-bound inference is dispatched to `tokio::task::spawn_blocking` to avoid starving the async runtime.

### Classification Prompt & Response Format

**Prompt** (injected as system instruction):
```
You are a request classifier for an AI Prompt Firewall. Analyse the user's prompt
and classify it into EXACTLY ONE of these categories:
- SIMPLE — Greetings, basic factual questions, short answers, simple math.
- COMPLEX — Deep reasoning, multi-step analysis, creative writing.
- RAG — Questions needing external documents or citations.
- CODEGEN — Code generation, debugging, programming tasks.

Reply with EXACTLY this format:
LABEL: <one of SIMPLE|COMPLEX|RAG|CODEGEN>
CONFIDENCE: <a number between 0.0 and 1.0>
```

**Expected response**:
```
LABEL: SIMPLE
CONFIDENCE: 0.95
```

The parser falls back to keyword detection and defaults to `COMPLEX` (safest routing) if the model produces unexpected output.

### Model Details

| Property | Value |
|---|---|
| **Base model** | Google Gemma-2-2B-IT |
| **Quantisation** | Q4_K_M (~1.5 GB) via GGUF |
| **HF repository** | `mradermacher/gemma-2-2b-it-GGUF` |
| **Tokenizer source** | `google/gemma-2-2b-it` (canonical) |
| **Candle backend** | `candle_transformers::models::quantized_llama::ModelWeights` |
| **Download** | Automatic via `hf-hub` (cached in `~/.cache/huggingface/`) |

## Configuration

| Env Variable | Field | Default | Used By |
|---|---|---|---|
| `ISARTOR__HOST_PORT` | `host_port` | `0.0.0.0:8080` | `main.rs` |
| `ISARTOR__GATEWAY_API_KEY` | `gateway_api_key` | `changeme` | Layer 0 |
| `ISARTOR__INFERENCE_ENGINE` | `inference_engine` | `sidecar` | Layer 2 engine mode |
| `ISARTOR__CACHE_BACKEND` | `cache_backend` | `memory` | `memory` (in-process LRU) or `redis` (distributed) |
| `ISARTOR__REDIS_URL` | `redis_url` | `redis://127.0.0.1:6379` | Redis connection (when `cache_backend=redis`) |
| `ISARTOR__ROUTER_BACKEND` | `router_backend` | `embedded` | `embedded` (Candle GGUF) or `vllm` (remote) |
| `ISARTOR__VLLM_URL` | `vllm_url` | `http://127.0.0.1:8000` | vLLM server URL (when `router_backend=vllm`) |
| `ISARTOR__VLLM_MODEL` | `vllm_model` | `gemma-2-2b-it` | vLLM model name (when `router_backend=vllm`) |
| `ISARTOR__LAYER2__SIDECAR_URL` | `layer2.sidecar_url` | `http://127.0.0.1:8081` | Layer 2 (sidecar mode) |
| `ISARTOR__LAYER2__MODEL_NAME` | `layer2.model_name` | `phi-3-mini` | Layer 2 (sidecar mode) |
| `ISARTOR__EMBEDDING_SIDECAR__SIDECAR_URL` | `embedding_sidecar.sidecar_url` | `http://127.0.0.1:8082` | v2 Pipeline (legacy; v1 uses in-process candle embedder) |
| `ISARTOR__EXTERNAL_LLM_API_KEY` | `external_llm_api_key` | *(empty)* | Layer 3 |

### Embedded Classifier Configuration

| Env Variable | Field | Default | Description |
|---|---|---|---|
| `ISARTOR__EMBEDDED_CLASSIFIER__REPO_ID` | `embedded_classifier.repo_id` | `mradermacher/gemma-2-2b-it-GGUF` | HF repository hosting the GGUF model |
| `ISARTOR__EMBEDDED_CLASSIFIER__GGUF_FILENAME` | `embedded_classifier.gguf_filename` | `gemma-2-2b-it.Q4_K_M.gguf` | GGUF model filename |
| `ISARTOR__EMBEDDED_CLASSIFIER__MAX_CLASSIFY_TOKENS` | `embedded_classifier.max_classify_tokens` | `20` | Max tokens for classification |
| `ISARTOR__EMBEDDED_CLASSIFIER__MAX_GENERATE_TOKENS` | `embedded_classifier.max_generate_tokens` | `256` | Max tokens for simple-task execution |
| `ISARTOR__EMBEDDED_CLASSIFIER__TEMPERATURE` | `embedded_classifier.temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `ISARTOR__EMBEDDED_CLASSIFIER__REPETITION_PENALTY` | `embedded_classifier.repetition_penalty` | `1.1` | Repetition penalty |

## Deployment

```mermaid
flowchart LR
    subgraph Build
        SRC["Source Code"] --> MUSL["Static Build\nx86_64-musl"]
    end

    subgraph Container
        MUSL --> BIN["/isartor\n~5 MB binary"]
        BIN --> IMG["distroless\nNo shell, no OS"]
    end

    subgraph Deploy
        IMG --> K8S["Kubernetes Pod"]
        IMG --> EDGE["Edge Node\nStandalone"]
    end

    subgraph Models["Model Cache"]
        HF["~/.cache/huggingface/\nGemma-2-2B-IT GGUF\n~1.5 GB\nall-MiniLM-L6-v2\n~90 MB"]
    end

    IMG -.->|"auto-download\non first start"| Models
```

### Memory Budget (Embedded Mode)

| Component | Approximate Size |
|---|---|
| Isartor binary | ~5 MB |
| candle embedding model (all-MiniLM-L6-v2) | ~90 MB |
| Gemma-2-2B-IT Q4_K_M | ~1.5 GB |
| Tokenizer | ~3 MB |
| Runtime overhead | ~50 MB |
| **Total** | **~1.65 GB** |

For VPS/edge deployments, a 2 GB RAM instance is sufficient. The GGUF model is memory-mapped where possible, so actual RSS may be lower than the file size. Model weights are downloaded once on first startup from Hugging Face Hub and cached in `~/.cache/huggingface/`.

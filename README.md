<p align="center">
  <img src="docs/logo.png" width="400" alt="Isartor Logo" />
</p>

# 🏛️ Isartor

**Algorithmic AI Orchestration Gateway**

A multi-layer intelligent proxy that classifies, caches, and routes AI traffic to the cheapest capable responder — built in Rust.

[Why Isartor](#why-isartor) •
[Architecture](#architecture) •
[Features](#key-features) •
[Deployment](#deployment-tiers) •
[Quick Start](#quick-start) •
[Configuration](#configuration) •
[Project Structure](#project-structure)

---

## Why Isartor?

Standard AI gateways are dumb pipes. Every prompt — trivial or complex — is forwarded blindly to heavyweight models like GPT-4o or Claude 3.5, burning through tokens and incurring unnecessary latency and cost.

**Isartor takes a different approach.** It implements a multi-layer algorithmic funnel that intercepts, classifies, and routes each request to the *cheapest capable responder*:

| Problem | Isartor's Answer |
| --- | --- |
| Identical prompts hitting the LLM repeatedly | **Semantic cache** returns instant answers at zero cost |
| Simple prompts consuming expensive tokens | **Embedded Gemma-2-2B classifier** resolves trivially in-process via candle |
| External sidecars add operational complexity | **Rust-native ML inference** — zero extra processes, zero network hops |
| No visibility into where cost is spent | **Per-layer OpenTelemetry tracing** shows exactly which layer resolved each request |
| Heavyweight runtimes on edge devices | **Single static binary (~5 MB)** runs on distroless containers or bare metal |
| One-size-fits-all deployment | **Three deployment tiers** — from a single binary on a €5 VPS to auto-scaling GPU pools on Kubernetes |

The result: only the hardest, most complex requests ever reach your expensive cloud LLM provider.

---

## Architecture

Isartor is a sequential processing pipeline built on [Axum](https://github.com/tokio-rs/axum) and [Tokio](https://tokio.rs/). Each layer can **short-circuit** the pipeline, returning a response directly and skipping all downstream work.

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                        Client Request                       │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Layer 0 — Operational Defense                              │
  │  Auth · Rate Limiting · Adaptive Concurrency Control        │
  │  Short-circuits: 401 Unauthorized / 503 Overloaded          │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Layer 1 — Semantic Cache                                   │
  │  Embed prompt → cosine similarity search → cache hit ⚡     │
  │  Short-circuits: returns cached answer (zero LLM cost)      │
  └──────────────────────────┬──────────────────────────────────┘
                             │  cache miss
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Layer 2 — Intelligent Triage                               │
  │  Embedded SLM or sidecar classifies intent                  │
  │  (Simple / Complex / RAG / CodeGen)                         │
  │  Short-circuits: simple tasks resolved locally 🟢           │
  └──────────────────────────┬──────────────────────────────────┘
                             │  complex / RAG
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Layer 2.5 — Context Optimiser                              │
  │  Retrieve candidate docs → rerank to top-K                  │
  │  Minimises token usage before the cloud call                │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Layer 3 — Cloud LLM Fallback                               │
  │  OpenAI · Azure OpenAI · Anthropic · xAI                    │
  │  Only the hardest prompts reach here 🔵                     │
  └─────────────────────────────────────────────────────────────┘
```

> **Detailed architecture documentation** — Mermaid diagrams, module map, embedded classifier internals, inference flow, and thread-safety model — lives in [`architecture.md`](architecture.md).

### Layer Summary

| Layer | Purpose | Backend |
| --- | --- | --- |
| **0 — Defense** | API key auth, adaptive concurrency limiter (AIMD) | `middleware/auth.rs` |
| **1 — Cache** | Embed prompt → cosine similarity → cache hit/miss | Embedding sidecar + in-memory vector store |
| **2 — Triage** | Classify intent, execute simple tasks locally | Embedded candle classifier *or* llama.cpp sidecar |
| **2.5 — Optimise** | Retrieve + rerank docs to top-K context | Reranker (sidecar or embedded) |
| **3 — Fallback** | Route to cloud LLM with optimised context | OpenAI / Azure / Anthropic / xAI via rig-core |

---

## Key Features

- **Rust-native performance** — Tokio + Axum async runtime; `opt-level = "z"` + LTO produces a ~5 MB static binary.
- **Multi-layer short-circuiting** — Each pipeline layer can resolve a request independently.
- **Embedded ML inference** — Gemma-2-2B-IT GGUF loaded in-process via [candle](https://github.com/huggingface/candle). Auto-downloads from Hugging Face on first start.
- **Three deployment tiers** — From a single binary on a VPS to auto-scaling GPU pools on Kubernetes.
- **Adaptive concurrency control** — AIMD-style limiter with P95 latency targets.
- **Semantic + exact caching** — Dual-mode cache with TTL expiry and capacity eviction.
- **Multi-provider LLM support** — OpenAI, Azure OpenAI, Anthropic, xAI via rig-core.
- **First-class OpenTelemetry** — Distributed tracing (Jaeger) and metrics (Prometheus/Grafana).
- **Configuration-driven** — `ISARTOR_*` env vars, TOML/YAML config files, or both.
- **Distroless container** — Multi-stage Docker build; final image has no shell, no OS.
- **Dual API surface** — v1 middleware pipeline (`/api/chat`) and v2 algorithmic pipeline (`/api/v2/chat`).

---

## Deployment Tiers

Isartor is designed to run at any scale. Because the gateway embeds ML models for local inference, we define **three distinct deployment strategies** based on your infrastructure:

```text
   Level 1                    Level 2                     Level 3
   Minimal                    Sidecar                     Enterprise
  ┌──────────┐            ┌──────────────────┐       ┌───────────────────┐
  │          │            │   Gateway Pod    │       │  Gateway Pods (N) │
  │ Isartor  │            │  ┌────────────┐  │       │   (stateless)     │
  │ (single  │            │  │  Isartor   │  │       └────────┬──────────┘
  │  binary) │            │  └─────┬──────┘  │                │
  │          │            │        │ HTTP     │                │ internal LB
  │ candle   │            │  ┌─────▼──────┐  │       ┌────────▼──────────┐
  │ embedded │            │  │ llama.cpp  │  │       │  Inference Pool   │
  │          │            │  │ sidecar    │  │       │  (vLLM / TGI)     │
  └──────────┘            │  └────────────┘  │       │  GPU auto-scale   │
                          └──────────────────┘       └───────────────────┘
  Target:                 Target:                    Target:
  VPS, Edge,              Docker Compose,            Kubernetes,
  docker run,             single host + GPU          Helm, HPA,
  bare metal                                         service mesh
```

| Tier | Strategy | Inference | Target | Guide |
| --- | --- | --- | --- | --- |
| **Level 1 — Minimal** | Monolithic static binary | In-process candle (Gemma-2-2B GGUF on CPU) | VPS, edge, `docker run`, bare metal | [docs/deploy-level1-minimal.md](docs/deploy-level1-minimal.md) |
| **Level 2 — Sidecar** | Split architecture | llama.cpp / TGI sidecar on same host via HTTP | Docker Compose, single host + GPU | [docs/deploy-level2-sidecar.md](docs/deploy-level2-sidecar.md) |
| **Level 3 — Enterprise** | Fully decoupled microservices | Auto-scaling GPU inference pools (vLLM, TGI) | Kubernetes, Helm, HPA | [docs/deploy-level3-enterprise.md](docs/deploy-level3-enterprise.md) |

> **All three tiers share the same binary and configuration surface.** The deployment tier is determined by environment variables and infrastructure, not code changes.

---

## Quick Start

Pick the path that matches your situation:

### 🟢 Fastest — Single Binary (Level 1)

```bash
git clone https://github.com/isartor-ai/isartor.git && cd isartor
cargo build --release
export ISARTOR_EXTERNAL_LLM_API_KEY="sk-..."
./target/release/isartor
# → Gateway on http://localhost:8080
# → Gemma-2-2B model auto-downloads on first start (~1.5 GB)
```

> **Full Level 1 guide** (systemd service, `docker run`, model pre-caching, memory tuning): [docs/deploy-level1-minimal.md](docs/deploy-level1-minimal.md)

### 🟡 Full Stack — Docker Compose (Level 2)

```bash
git clone https://github.com/isartor-ai/isartor.git && cd isartor/docker
docker compose -f docker-compose.full.yml up --build
```

Once the containers are healthy:

| Service | URL | Purpose |
| --- | --- | --- |
| **Isartor Gateway** | http://localhost:8080 | AI Gateway (v1 + v2 endpoints) |
| **Jaeger UI** | http://localhost:16686 | Distributed tracing |
| **Grafana** | http://localhost:3000 | Metrics dashboards |
| **Prometheus** | http://localhost:9090 | Metrics storage |
| **SLM Generation** | http://localhost:8081 | llama.cpp — Phi-3-mini |
| **SLM Embedding** | http://localhost:8082 | llama.cpp — all-MiniLM-L6-v2 |

> **Full Level 2 guide** (GPU passthrough, sidecar configuration, scaling): [docs/deploy-level2-sidecar.md](docs/deploy-level2-sidecar.md)

### 🔵 Enterprise — Kubernetes (Level 3)

> **Full Level 3 guide** (Helm charts, HPA, inference pools, ingress, service mesh): [docs/deploy-level3-enterprise.md](docs/deploy-level3-enterprise.md)

### Send a Test Request

```bash
curl -s http://localhost:8080/api/v2/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{"prompt": "What is the capital of France?"}' | jq .
```

```json
{
  "resolved_by_layer": 2,
  "message": "The capital of France is Paris.",
  "model": "gemma-2-2b-it",
  "total_duration_ms": 142,
  "processing_log": [
    { "step": "Layer0_AdaptiveConcurrency", "terminal": false, "duration_ms": 0 },
    { "step": "Layer1_SemanticCache", "detail": "Cache MISS", "terminal": false },
    { "step": "Layer2_IntentClassifier", "detail": "Classified as Simple (confidence=0.940)" },
    { "step": "Layer2_LocalExecutor", "detail": "Simple task executed by local SLM", "terminal": true }
  ]
}
```

### Health Check

```bash
curl http://localhost:8080/healthz
# {"status":"ok"}
```

---

## Configuration

All settings are controlled via environment variables prefixed with `ISARTOR_`, an optional `isartor.toml` / `isartor.yaml` config file, or both (env vars take precedence).

### Key Variables (Summary)

| Variable | Default | Description |
| --- | --- | --- |
| `ISARTOR_HOST_PORT` | `0.0.0.0:8080` | Listen address |
| `ISARTOR_GATEWAY_API_KEY` | `changeme` | API key for `X-API-Key` auth |
| `ISARTOR_CACHE_MODE` | `both` | `exact`, `semantic`, or `both` |
| `ISARTOR_LLM_PROVIDER` | `openai` | `openai` / `azure` / `anthropic` / `xai` |
| `ISARTOR_EXTERNAL_LLM_API_KEY` | *(empty)* | Cloud LLM provider API key |
| `ISARTOR_ENABLE_MONITORING` | `false` | Enable OpenTelemetry export |

> **Full configuration reference** with all env vars, TOML examples, and per-tier defaults: [docs/configuration.md](docs/configuration.md)

---

## Observability

Isartor ships with first-class [OpenTelemetry](https://opentelemetry.io/) support. When `ISARTOR_ENABLE_MONITORING=true`, the gateway exports distributed traces and metrics.

| Service | URL | Purpose |
| --- | --- | --- |
| **Jaeger UI** | http://localhost:16686 | Distributed tracing |
| **Grafana** | http://localhost:3000 | Metrics dashboards |
| **Prometheus** | http://localhost:9090 | Metrics storage |
| **OTel Collector** | localhost:4317 (gRPC) | Telemetry pipeline |

> **Full observability guide** with per-tier setup, custom dashboards, and alerting: [docs/observability.md](docs/observability.md)

---

## Project Structure

```text
src/
├── main.rs                          # Bootstrap, router, middleware wiring
├── config.rs                        # AppConfig — env vars + file loading
├── state.rs                         # Shared AppState, rig-core LLM agents
├── handler.rs                       # v1 Layer 3 fallback handler
├── models.rs                        # Request / response types
├── telemetry.rs                     # OpenTelemetry initialisation
├── vector_cache.rs                  # HNSW-backed vector cache
├── middleware/
│   ├── auth.rs                      # Layer 0 — API key authentication
│   ├── cache.rs                     # Layer 1 — Semantic + exact cache
│   ├── slm_triage.rs                # Layer 2 — SLM intent classification
│   └── monitoring.rs                # Root tracing middleware
├── services/
│   └── local_inference.rs           # Embedded Classifier (candle + Gemma-2-2B-IT)
└── pipeline/                        # v2 Algorithmic Pipeline
    ├── context.rs                   # PipelineContext, IntentClassification
    ├── traits.rs                    # Embedder, VectorStore, Reranker, …
    ├── orchestrator.rs              # Pipeline execution engine
    ├── concurrency.rs               # Adaptive concurrency limiter
    └── implementations/
        ├── embedder.rs              # LlamaCppEmbedder (Layer 1)
        ├── vector_store.rs          # InMemoryVectorStore (Layer 1)
        ├── intent_classifier.rs     # LlamaCppIntentClassifier (Layer 2)
        ├── local_executor.rs        # LlamaCppLocalExecutor (Layer 2)
        ├── reranker.rs              # LlamaCppReranker (Layer 2.5)
        └── external_llm.rs          # RigExternalLlm (Layer 3)

docs/                                # Tiered deployment documentation
├── deploy-level1-minimal.md         # Level 1: Edge / VPS deployment
├── deploy-level2-sidecar.md         # Level 2: Docker Compose + sidecar
├── deploy-level3-enterprise.md      # Level 3: Kubernetes / microservices
├── configuration.md                 # Full configuration reference
├── observability.md                 # OTel, Jaeger, Prometheus, Grafana
└── architecture-decisions.md        # ADRs and design rationale

docker/
├── Dockerfile                       # Multi-stage static musl build
├── Dockerfile.minimal               # Level 1: distroless single-binary image
├── Dockerfile.sidecar               # Level 2: debian-slim gateway image
├── docker-compose.full.yml          # Full stack (Level 2 + observability)
├── docker-compose.sidecar.yml       # Level 2: gateway + inference sidecar
├── docker-compose.yml               # Gateway + sidecars (OpenAI)
├── docker-compose.azure.yml         # Gateway + sidecars (Azure)
├── docker-compose.observability.yml # Observability-only stack
├── .env.example                     # Environment template (legacy)
├── .env.full.example                # Environment template (full stack)
├── .env.sidecar.example             # Environment template (Level 2 sidecar)
├── otel-collector-config.yaml       # OTel Collector pipeline
└── prometheus.yml                   # Prometheus scrape targets
```

---

## Documentation Index

| Document | Description |
| --- | --- |
| [README.md](README.md) | This file — project overview and quick start |
| [architecture.md](architecture.md) | Detailed architecture: Mermaid diagrams, module map, embedded classifier internals |
| [docs/deploy-level1-minimal.md](docs/deploy-level1-minimal.md) | **Level 1** — Edge/VPS: static binary, embedded candle, systemd, `docker run` |
| [docs/deploy-level2-sidecar.md](docs/deploy-level2-sidecar.md) | **Level 2** — Mid-tier: Docker Compose, llama.cpp/TGI sidecar, GPU passthrough |
| [docs/deploy-level3-enterprise.md](docs/deploy-level3-enterprise.md) | **Level 3** — Enterprise: Kubernetes, Helm, HPA, inference pools, service mesh |
| [docs/configuration.md](docs/configuration.md) | Full configuration reference: all env vars, TOML examples, per-tier defaults |
| [docs/observability.md](docs/observability.md) | Observability guide: OTel setup, Jaeger, Prometheus, Grafana, alerting |
| [docs/architecture-decisions.md](docs/architecture-decisions.md) | Architecture Decision Records (ADRs) and design rationale |

---

## License

*TBD — add your licence here.*

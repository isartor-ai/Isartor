# Isartor

**An ultra-lightweight, pure-Rust Smart Algorithm Gateway designed to execute local intelligence, slash LLM costs, and accelerate agentic workloads.**

<p align="center">
  <img src="docs/logo.png" alt="Isartor" width="400">
</p>

[![CI](https://github.com/isartor-ai/Isartor/actions/workflows/ci.yml/badge.svg)](https://github.com/isartor-ai/Isartor/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/isartor-ai/Isartor/releases)

---

## The Problem: "Dumb Pipes" for Smart Workloads

Standard API gateways are "dumb pipes" — they treat AI prompts like standard HTTP traffic, blindly forwarding every request to cloud LLMs (GPT-4, Claude) regardless of complexity.

For autonomous agents and enterprise applications, this is a fatal flaw. Agent loops repeat identical prompts. Simple data extraction tasks consume the same expensive tokens as complex reasoning problems. The result is runaway costs, high latency, and sensitive data leaving your perimeter unnecessarily.

## The Solution: A Smart Algorithm Gateway

Isartor replaces the dumb pipe with algorithmic intelligence at the edge. Acting as a drop-in OpenAI replacement, it intercepts incoming prompts and applies a cascade of local algorithms — from deterministic hashing to pure-Rust neural networks — to resolve requests locally.

By computing intent *before* routing, Isartor acts as an impenetrable financial shield for your LLM spend.

- **100% Pure-Rust Edge AI:** Statically compiled, no dependency hell. Native tensor execution via Hugging Face `candle`.
- **Algorithmic Deflection:** Layers 1 & 2 can resolve 60–80% of repetitive agentic traffic locally for $0.
- **Frictionless:** One `cargo build` or `docker run` and you're live.

---

## The Algorithmic Funnel (Architecture)

Every incoming request passes through a sequence of smart computing layers. Only prompts requiring genuine, complex reasoning survive the funnel to reach the cloud.

```text
Request ──► L1a Deterministic Hash ──► L1b Vector Search ──► L2 Neural Router ──► L3 Cloud Fallback
                   │ hit                    │ hit                 │ simple               │
                   ▼                        ▼                     ▼                      ▼
                Response                 Response             Local Response         Cloud Response
```

| Layer | Algorithm | What It Does | Typical Latency |
|:------|:----------|:-------------|:----------------|
| **L1a — Deterministic Hash** | `ahash` + LRU cache | Catches exact-duplicate requests (e.g. agent loops) with sub-millisecond lookup | < 1 ms |
| **L1b — Vector Search** | `candle` BertModel (`all-MiniLM-L6-v2`) + brute-force cosine similarity | Catches meaning-based duplicates ("What's the price?" ≈ "How much?") via pure-Rust local embeddings | 1–5 ms |
| **L2 — Neural Router** | Embedded SLM (Qwen-1.5B via `candle` GGUF) | Classifies intent and resolves trivial prompts in-process — no cloud round-trip | 50–200 ms |
| **L3 — Cloud Fallback** | OpenAI, Anthropic, Azure OpenAI, xAI | Forwards genuinely complex reasoning to the configured cloud provider | Network-bound |

Layers 1a and 1b alone can deflect **60–80% of agentic traffic** before any neural inference runs.

---

## Minimalist to Enterprise

Isartor uses a **Pluggable Trait Provider** pattern (Hexagonal Architecture). The same compiled binary adapts from a developer laptop to a multi-replica Kubernetes cluster. Switch modes entirely through environment variables — no code changes, no recompilation.

| Component | Minimalist (Single Binary) | Enterprise (K8s) |
|:----------|:---------------------------|:------------------|
| **L1a Hash Cache** | In-memory LRU (`ahash` + `parking_lot`) | Redis cluster (shared across replicas) |
| **L1b Vector Search** | In-process `candle` BertModel | External TEI sidecar (optional) |
| **L2 Neural Router** | Embedded `candle` GGUF inference | Remote vLLM / TGI server (GPU pool) |
| **L3 Cloud** | Direct to OpenAI / Anthropic | Direct to OpenAI / Anthropic |

**Minimalist Mode** — zero external dependencies. Download the binary and run it.

**Enterprise Mode** — set a few environment variables:

```bash
export ISARTOR__CACHE_BACKEND=redis
export ISARTOR__REDIS_URL=redis://redis-cluster.svc:6379
export ISARTOR__ROUTER_BACKEND=vllm
export ISARTOR__VLLM_URL=http://vllm.svc:8000
export ISARTOR__VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
```

---

## Quick Start

### Docker (Recommended)

All required ML models are baked into the image.

```bash
docker run -p 3000:3000 ghcr.io/isartor-ai/isartor:latest
```

### macOS / Linux

```bash
curl -fsSL https://raw.githubusercontent.com/isartor-ai/isartor/main/scripts/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/isartor-ai/isartor/main/scripts/install.ps1 | iex
```

### Build from Source

```bash
git clone https://github.com/isartor-ai/Isartor.git
cd Isartor
cargo build --release
./target/release/isartor
```

### Verify

```bash
curl -X POST http://localhost:3000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Calculate 2+2"}'
```

---

## Drop-In Integration

Isartor exposes an OpenAI-compatible API. Point any SDK or agent at it by changing a single URL.

```python
import openai

client = openai.OpenAI(base_url="http://localhost:3000/v1", api_key="your-gateway-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Summarise this document."}],
)
```

This works with **any** OpenAI-compatible client — the official Python/Node SDKs, LangChain, LlamaIndex, AutoGen, or autonomous agents like [OpenClaw](https://github.com/isartor-ai/openclaw). No code changes beyond the base URL.

---

## Enterprise Observability

Isartor emits standard **OpenTelemetry** traces and metrics out of the box.

- **Distributed traces** — every request produces a root span (`gateway_request`) with child spans for each layer (`l1a_exact_cache`, `l1b_semantic_cache`, `l2_classify_intent`, `l3_cloud_llm`).
- **Prometheus metrics** — `isartor_request_duration_seconds`, `isartor_layer_duration_seconds`, `isartor_requests_total`.
- **ROI metric** — `isartor_tokens_saved_total` tracks estimated tokens that never left your infrastructure. Pipe it into Grafana to prove cost savings to leadership.

Enable with:

```bash
export ISARTOR__ENABLE_MONITORING=true
export ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317
```

See [docs/6-OBSERVABILITY.md](docs/6-OBSERVABILITY.md) for the full span and metric reference.

---

## Documentation

| Guide | Description |
|:------|:------------|
| [Quick Start](docs/1-QUICKSTART.md) | Installation, first request, configuration basics |
| [Architecture](docs/2-ARCHITECTURE.md) | Deep dive into the 3-layer funnel and trait provider pattern |
| [Enterprise Guide](docs/3-ENTERPRISE-GUIDE.md) | Redis, vLLM, Kubernetes, Helm, horizontal scaling |
| [Integrations](docs/4-INTEGRATIONS.md) | OpenAI SDK, LangChain, autonomous agents |
| [Configuration Reference](docs/5-CONFIGURATION-REF.md) | Every environment variable and config key |
| [Observability](docs/6-OBSERVABILITY.md) | OpenTelemetry spans, metrics, Grafana dashboards |

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

# Isartor

**An ultra-lightweight, pure-Rust Prompt Firewall designed to execute local intelligence, slash LLM costs, and accelerate agentic workloads.**

<p align="center">
  <img src="docs/logo.png" alt="Isartor" width="400">
</p>

[![CI](https://github.com/isartor-ai/Isartor/actions/workflows/ci.yml/badge.svg)](https://github.com/isartor-ai/Isartor/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/isartor-ai/Isartor/releases)
[![Air-Gappable](https://img.shields.io/badge/%E2%9C%93%20Air--Gappable-FedRAMP%20Ready-blue)](docs/AIR-GAPPED.md)
[![Zero Telemetry](https://img.shields.io/badge/%E2%9C%93%20Zero%20hidden%20telemetry-verified%20by%20CI-brightgreen)](tests/phone_home_audit.rs)

---

## The Problem: "Dumb Pipes" for Smart Workloads

Standard API gateways are "dumb pipes" — they treat AI prompts like standard HTTP traffic, blindly forwarding every request to cloud LLMs (GPT-4, Claude) regardless of complexity.

For autonomous agents and enterprise applications, this is a fatal flaw. Agent loops repeat identical prompts. Simple data extraction tasks consume the same expensive tokens as complex reasoning problems. The result is runaway costs, high latency, and sensitive data leaving your perimeter unnecessarily.

## The Solution: A Prompt Firewall

Isartor replaces the dumb pipe with algorithmic intelligence at the edge. Acting as a drop-in OpenAI replacement, it intercepts incoming prompts and applies a cascade of local algorithms — from deterministic hashing to pure-Rust neural networks — to resolve requests locally.

By computing intent *before* routing, Isartor acts as an impenetrable Prompt Firewall for your LLM spend.

- **100% Pure-Rust Edge AI:** Statically compiled, no dependency hell. Native tensor execution via Hugging Face `candle`.
- **Algorithmic Deflection:** In our benchmark suite, L1a and L1b deflect **71% of repetitive agentic traffic** (FAQ/agent loop patterns) and **38% of diverse task traffic**. [Full benchmark →](benchmarks/README.md)
- **Frictionless:** One `cargo build` or `docker run` and you're live.

---

## The Deflection Stack (Architecture)

Every incoming request passes through a sequence of smart computing layers. Only prompts requiring genuine, complex reasoning survive the Deflection Stack to reach the cloud.

```text
Request ──► L1a Exact Cache ──► L1b Semantic Cache ──► L2 SLM Router ──► L2.5 Context Optimiser ──► L3 Cloud Logic
                 │ hit                │ hit                 │ simple             │ compressed                │
                 ▼                    ▼                     ▼                    ▼                           ▼
              Response             Response            Local Response     Optimised Prompt            Cloud Response
```

| Layer | Algorithm / Mechanism | What It Does | Typical Latency |
|:------|:----------------------|:-------------|:----------------|
| **L1a — Exact Cache** | Fast Hashing (`ahash`) | Sub-millisecond duplicate detection. Traps infinite agent loops instantly. | < 1 ms |
| **L1b — Semantic Cache** | Cosine Similarity (Embeddings) | Computes mathematical meaning via pure-Rust `candle` models (`all-MiniLM-L6-v2`) to catch variations ("Price?" ≈ "Cost?"). | 1–5 ms |
| **L2 — SLM Router** | Neural Classification (LLM) | Triages intent using an embedded Small Language Model (e.g. Qwen-1.5B) to resolve simple data extraction tasks. | 50–200 ms |
| **L2.5 — Context Optimiser** | Retrieve + Rerank (top-K) | Retrieves and reranks candidate documents to minimise token usage before the cloud call. | 5–50 ms |
| **L3 — Cloud Logic** | Load Balancing & Retries | Routes surviving complex prompts to OpenAI, Anthropic, or Azure, with built-in fallback resilience. | Network-bound |

Layers 1a and 1b deflect **71% of repetitive agentic traffic** (FAQ/agent loop patterns) and **38% of diverse task traffic** before any neural inference runs. [Full benchmark →](benchmarks/README.md)

---

## Minimalist to Enterprise

Isartor uses a **Pluggable Trait Provider** pattern (Hexagonal Architecture). The same compiled binary adapts from a developer laptop to a multi-replica Kubernetes cluster. Switch modes entirely through environment variables — no code changes, no recompilation.

| Component | Minimalist (Single Binary) | Enterprise (K8s) |
|:----------|:---------------------------|:------------------|
| **L1a Exact Cache** | In-memory LRU (`ahash` + `parking_lot`) | Redis cluster (shared across replicas) |
| **L1b Semantic Cache** | In-process `candle` BertModel | External TEI sidecar (optional) |
| **L2 SLM Router** | Embedded `candle` GGUF inference | Remote vLLM / TGI server (GPU pool) |
| **L2.5 Context Optimiser** | In-process retrieve + rerank (top-K selection) | Distributed rerank (optional TEI / ANN pool) |
| **L3 Cloud Logic** | Direct to OpenAI / Anthropic | Direct to OpenAI / Anthropic |

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

### Docker (Recommended — minimal configuration)

The image ships a statically linked `isartor` binary and downloads the embedding model on first start (then reuses the on-disk hf-hub cache). No API key is needed for the cache layers.

```bash
docker run -p 8080:8080 ghcr.io/isartor-ai/isartor:latest
```

To persist the model cache across restarts (recommended):

```bash
docker run -p 8080:8080 \
  -e HF_HOME=/tmp/huggingface \
  -v isartor-hf:/tmp/huggingface \
  ghcr.io/isartor-ai/isartor:latest
```

The startup banner appears after all layers are ready (< 30 s on a modern machine).
Verify with:

```bash
curl http://localhost:8080/health
# {"status":"ok","version":"0.1.0","layers":{...},"uptime_seconds":5,"demo_mode":true}
```

> **Image size:** ~120 MB compressed / ~260 MB on disk (includes `all-MiniLM-L6-v2` embedding model, statically linked Rust binary).

### macOS / Linux — single command

```bash
curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh
```

After installation:

```bash
isartor          # start the server on port 8080
isartor demo     # run the deflection demo (no API key needed)
isartor init     # generate a commented config scaffold
```

### Windows (PowerShell) — single command

```powershell
irm https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.ps1 | iex
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
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Calculate 2+2"}'
```

---

## Drop-In Integration

Isartor exposes an OpenAI-compatible API. Point any SDK or agent at it by changing a single URL.

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="your-api-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Summarise this document."}],
)
```

This works with **any** OpenAI-compatible client — the official Python/Node SDKs, LangChain, LlamaIndex, AutoGen, or autonomous agents like [OpenClaw](https://github.com/isartor-ai/openclaw). No code changes beyond the base URL.

---

## Enterprise Observability

Isartor emits standard **OpenTelemetry** traces and metrics out of the box.

- **Distributed traces** — every request produces a root span (`gateway_request`) with child spans for each layer (`l1a_exact_cache`, `l1b_semantic_cache`, `l2_classify_intent`, `context_optimise`, `l3_cloud_llm`).
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
| [Architecture](docs/2-ARCHITECTURE.md) | Deep dive into the Deflection Stack and trait provider pattern |
| [Enterprise Guide](docs/3-ENTERPRISE-GUIDE.md) | Redis, vLLM, Kubernetes, Helm, horizontal scaling |
| [Integrations](docs/4-INTEGRATIONS.md) | OpenAI SDK, LangChain, autonomous agents |
| [Configuration Reference](docs/5-CONFIGURATION-REF.md) | Every environment variable and config key |
| [Observability](docs/6-OBSERVABILITY.md) | OpenTelemetry spans, metrics, Grafana dashboards |
| [Performance Tuning](docs/PERFORMANCE-TUNING.md) | Deflection measurement, config tuning, SLO/SLA templates |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues, diagnostic steps, FAQ |
| [Governance](GOVERNANCE.md) | Project independence, license stability, decision making |

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

# Isartor

**An ultra-lightweight, pure-Rust Prompt Firewall designed to execute local intelligence, slash LLM costs, and accelerate agentic workloads.**

<p align="center">
  <img src="docs/logo.png" alt="Isartor" width="400">
</p>

[![CI](https://github.com/isartor-ai/Isartor/actions/workflows/ci.yml/badge.svg)](https://github.com/isartor-ai/Isartor/actions)
[![codecov](https://codecov.io/gh/isartor-ai/Isartor/branch/main/graph/badge.svg)](https://codecov.io/gh/isartor-ai/Isartor)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/isartor-ai/Isartor?display_name=tag&sort=semver)](https://github.com/isartor-ai/Isartor/releases/latest)
[![Docs](https://img.shields.io/badge/docs-isartor--ai.github.io-blue)](https://isartor-ai.github.io/Isartor/)
[![Air-Gappable](https://img.shields.io/badge/%E2%9C%93%20Air--Gappable-FedRAMP%20Ready-blue)](https://isartor-ai.github.io/Isartor/deployment/air-gapped.html)
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
- **AI Tool Ready:** Works out of the box with GitHub Copilot, Cursor, Claude Code, OpenAI Codex CLI, Gemini CLI, and any OpenAI-compatible tool.

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

### macOS / Linux — single command (Recommended — minimal configuration)

```bash
curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh
```

### Docker (Straight forward — minimal configuration)

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

To use **Azure OpenAI** for Layer 3 (recommended: Docker secrets via `*_FILE`). Important: `ISARTOR__EXTERNAL_LLM_URL` must be the **base Azure endpoint only** (no `/openai/...` path), e.g. `https://<resource>.openai.azure.com`:

```bash
# Put your key in a file (no trailing newline is ideal, but Isartor trims whitespace)
echo -n "YOUR_AZURE_OPENAI_KEY" > ./azure_openai_key

docker run -p 8080:8080 \
  -e ISARTOR__LLM_PROVIDER=azure \
  -e ISARTOR__EXTERNAL_LLM_URL=https://<resource>.openai.azure.com \
  -e ISARTOR__AZURE_DEPLOYMENT_ID=<deployment> \
  -e ISARTOR__AZURE_API_VERSION=2024-08-01-preview \
  -e ISARTOR__EXTERNAL_LLM_API_KEY_FILE=/run/secrets/azure_openai_key \
  -v $(pwd)/azure_openai_key:/run/secrets/azure_openai_key:ro \
  ghcr.io/isartor-ai/isartor:latest
```

The startup banner appears after all layers are ready (< 30 s on a modern machine).
Verify with:

```bash
curl http://localhost:8080/health
# {"status":"ok","version":"0.1.42","layers":{...},"uptime_seconds":5,"demo_mode":true}
```

> **Image size:** ~120 MB compressed / ~260 MB on disk (includes `all-MiniLM-L6-v2` embedding model, statically linked Rust binary).



After installation:

```bash
isartor up                    # start the API gateway
isartor up --detach           # start in background
isartor up copilot            # start gateway + Copilot CONNECT proxy
isartor demo                  # run the deflection demo (no API key needed)
isartor init                  # generate a commented config scaffold
isartor set-key -p openai     # configure your LLM provider API key
isartor connect copilot       # connect GitHub Copilot CLI via MCP
isartor connect copilot-vscode # configure GitHub Copilot in VS Code
isartor connect claude-copilot # use Claude Code with GitHub Copilot through Isartor
isartor connect cursor        # connect Cursor IDE
isartor connect claude        # connect Claude Code
isartor connect codex         # connect OpenAI Codex CLI
isartor connect gemini        # connect Gemini CLI
isartor connect opencode      # connect OpenCode
isartor stats                 # prompt totals, layer hits, routing history
isartor stats --by-tool       # per-tool requests, cache safety, latency, retries, errors
isartor stop                  # stop a running Isartor instance
isartor update                # self-update to the latest release
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
./target/release/isartor up
```

### Verify

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Calculate 2+2"}'
```

---

## AI Tool Integrations

Connect your favourite AI coding tools to Isartor with a single command. Isartor caches repetitive prompts so your tools respond faster and cost less.

| Tool | Connect Command | How It Works |
|:-----|:----------------|:-------------|
| **GitHub Copilot CLI** | `isartor connect copilot` | MCP tools (`isartor_chat` / `isartor_cache_store`) via stdio or `/mcp/` HTTP/SSE |
| **GitHub Copilot in VS Code** | `isartor connect copilot-vscode` | Managed `settings.json` debug overrides |
| **Claude Code + GitHub Copilot** | `isartor connect claude-copilot` | Claude base URL override + Copilot-backed Layer 3 |
| **Cursor IDE** | `isartor connect cursor` | Base URL override + HTTP/SSE MCP registration at `/mcp/` |
| **Claude Code** | `isartor connect claude` | `ANTHROPIC_BASE_URL` env override |
| **OpenAI Codex CLI** | `isartor connect codex` | `OPENAI_BASE_URL` env script |
| **Gemini CLI** | `isartor connect gemini` | `GEMINI_API_BASE_URL` env script |
| **OpenCode** | `isartor connect opencode` | Global OpenCode provider + auth config |
| **Any OpenAI-compatible tool** | `isartor connect generic` | Configurable env var override |

See the [full integration guides →](https://isartor-ai.github.io/Isartor/integrations/overview.html)

### Use Claude Code with GitHub Copilot (experimental)

```bash
# Requires: active GitHub Copilot subscription
# Default auth path: GitHub browser/device-flow login
isartor connect claude-copilot
isartor stop
isartor up --detach
claude
```

One-click smoke test:

```bash
./scripts/claude-copilot-smoke-test.sh
# or: make smoke-claude-copilot
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

See the [Observability guide →](https://isartor-ai.github.io/Isartor/observability/metrics-tracing.html) for the full span and metric reference.

---

## Documentation

📚 **Full docs site: [isartor-ai.github.io/Isartor](https://isartor-ai.github.io/Isartor/)**

| Guide | Description |
|:------|:------------|
| [Getting Started](https://isartor-ai.github.io/Isartor/getting-started/installation.html) | Installation, first request, configuration basics |
| [Architecture](https://isartor-ai.github.io/Isartor/concepts/architecture.html) | Deep dive into the Deflection Stack and trait provider pattern |
| [AI Tool Integrations](https://isartor-ai.github.io/Isartor/integrations/overview.html) | Copilot, Cursor, Claude Code, Codex, Gemini CLI, generic |
| [Deployment](https://isartor-ai.github.io/Isartor/deployment/level1-minimal.html) | Minimal, Sidecar (Docker Compose), Enterprise (K8s), Air-Gapped |
| [Configuration Reference](https://isartor-ai.github.io/Isartor/configuration/reference.html) | Every environment variable and config key |
| [Observability](https://isartor-ai.github.io/Isartor/observability/metrics-tracing.html) | OpenTelemetry spans, metrics, Grafana dashboards |
| [Performance Tuning](https://isartor-ai.github.io/Isartor/observability/performance-tuning.html) | Deflection measurement, config tuning, SLO/SLA templates |
| [Troubleshooting](https://isartor-ai.github.io/Isartor/development/troubleshooting.html) | Common issues, diagnostic steps, FAQ |
| [Contributing](https://isartor-ai.github.io/Isartor/development/contributing.html) | How to contribute, development setup, PR guidelines |
| [Governance](GOVERNANCE.md) | Project independence, license stability, decision making |

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

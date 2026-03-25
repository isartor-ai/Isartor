# Welcome to Isartor

<p align="center">
  <img src="logo.png" alt="Isartor" width="300">
</p>

**Open-source Prompt Firewall — deflect up to 95% of redundant LLM traffic before it leaves your infrastructure.**

Pure Rust · Single Binary · Zero Hidden Telemetry · Air-Gappable

---

AI coding agents and personal assistants repeat themselves — a lot. Copilot, Claude Code, Cursor, and OpenClaw send the same system instructions, the same context preambles, and often the same user prompts across every turn. Standard API gateways forward all of it to cloud LLMs regardless.

Isartor sits between your tools and the cloud. It intercepts every prompt and runs a cascade of local algorithms — from sub-millisecond hashing to in-process neural inference — to resolve requests before they reach the network. Only the genuinely hard prompts make it through.

## The Deflection Stack

Every incoming request passes through a sequence of smart computing layers. Only prompts requiring genuine, complex reasoning survive the stack to reach the cloud.

```text
Request ──► L1a Exact Cache ──► L1b Semantic Cache ──► L2 SLM Router ──► L2.5 Context Optimiser ──► L3 Cloud Logic
                 │ hit                │ hit                 │ simple             │ compressed                │
                 ▼                    ▼                     ▼                    ▼                           ▼
              Response             Response            Local Response     Optimised Prompt            Cloud Response
```

| Layer | What It Does | Typical Latency |
|:------|:-------------|:----------------|
| **L1a — Exact Cache** | Sub-millisecond duplicate detection via fast hashing. Traps infinite agent loops instantly. | < 1 ms |
| **L1b — Semantic Cache** | Catches meaning-equivalent prompts ("Price?" ≈ "Cost?") using pure-Rust embeddings. | 1–5 ms |
| **L2 — SLM Router** | Triages intent with an embedded Small Language Model to resolve simple tasks locally. | 50–200 ms |
| **L2.5 — Context Optimiser** | Compresses repeated instruction payloads (CLAUDE.md, copilot-instructions) via session dedup and minification. | < 1 ms |
| **L3 — Cloud Logic** | Routes surviving complex prompts to OpenAI, Anthropic, or Azure with fallback resilience. | Network-bound |

Layers 1a and 1b deflect **71% of repetitive agentic traffic** and **38% of diverse task traffic** before any neural inference runs.

## How It Works

Getting started with Isartor takes three steps:

### 1. Install

```bash
curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh
```

Or use Docker:

```bash
docker run -p 8080:8080 ghcr.io/isartor-ai/isartor:latest
```

### 2. Connect

Point any OpenAI-compatible client at Isartor — just change the base URL:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-api-key",
)
```

Works with the official SDKs, LangChain, LlamaIndex, AutoGen, GitHub Copilot, OpenClaw, and any other OpenAI-compatible tool.

Recent OpenAI-compatible improvements for coding agents include:

- `GET /v1/models` for model discovery
- `stream: true` support on `/v1/chat/completions` with proper SSE chunks
- `tools`, `tool_choice`, `functions`, and `function_call` passthrough
- `tool_calls` preserved in upstream responses

### 3. Save

Isartor deflects repetitive and simple prompts locally. You keep the same responses, pay for fewer tokens, and get lower latency — with zero code changes beyond the URL.

---

## Explore the Docs

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5em 2em;">

**[🚀 Getting Started](getting-started/installation.md)**
Install Isartor and send your first request.

**[🔌 Integrations](integrations/overview.md)**
Connect Copilot CLI, Cursor, Claude Code, and more.

**[📦 Deployment](deployment/level1-minimal.md)**
From a single binary to a multi-replica K8s cluster.

**[⚙️ Configuration](configuration/reference.md)**
Every environment variable and config key.

**[🏗️ Architecture](concepts/architecture.md)**
Deep dive into the Deflection Stack and trait providers.

**[📊 Observability](observability/metrics-tracing.md)**
OpenTelemetry traces, Prometheus metrics, Grafana dashboards.

</div>

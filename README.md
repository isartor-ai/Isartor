<p align="center">
  <img src="docs/logo.png" alt="Isartor" width="400">
</p>

<h1 align="center">Isartor</h1>

<p align="center">
  <strong>Open-source Prompt Firewall — deflect up to 95% of redundant LLM traffic before it leaves your infrastructure.</strong>
</p>

<p align="center">
  Pure Rust · Single Binary · Zero Hidden Telemetry · Air-Gappable
</p>

<p align="center">
  <a href="https://github.com/isartor-ai/Isartor/actions"><img src="https://github.com/isartor-ai/Isartor/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://codecov.io/gh/isartor-ai/Isartor"><img src="https://codecov.io/gh/isartor-ai/Isartor/branch/main/graph/badge.svg" alt="codecov" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" /></a>
  <a href="https://github.com/isartor-ai/Isartor/releases/latest"><img src="https://img.shields.io/github/v/release/isartor-ai/Isartor?display_name=tag&sort=semver" alt="Release" /></a>
  <a href="https://isartor-ai.github.io/Isartor/"><img src="https://img.shields.io/badge/docs-isartor--ai.github.io-blue" alt="Docs" /></a>
</p>

---

## Quick Start

```bash
# Install (macOS / Linux)
curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh

# Start the gateway
isartor up

# Connect your AI tool (pick one)
isartor connect copilot          # GitHub Copilot CLI
isartor connect claude           # Claude Code
isartor connect cursor           # Cursor IDE
isartor connect openclaw         # OpenClaw
isartor connect codex            # OpenAI Codex CLI
isartor connect gemini           # Gemini CLI
isartor connect claude-copilot   # Claude Code + GitHub Copilot
```

That's it. Your tool now routes through Isartor and every repeated prompt is resolved locally in under 1 ms.

<details>
<summary><strong>More install options</strong> (Docker · Windows · Build from source)</summary>

#### Docker

```bash
docker run -p 8080:8080 \
  -e HF_HOME=/tmp/huggingface \
  -v isartor-hf:/tmp/huggingface \
  ghcr.io/isartor-ai/isartor:latest
```

> ~120 MB compressed. Includes the `all-MiniLM-L6-v2` embedding model and a statically linked Rust binary.

#### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.ps1 | iex
```

#### Build from source

```bash
git clone https://github.com/isartor-ai/Isartor.git
cd Isartor && cargo build --release
./target/release/isartor up
```

</details>

---

## Why Isartor?

AI coding agents and personal assistants repeat themselves — a lot. Copilot, Claude Code, Cursor, and OpenClaw send the same system instructions, the same context preambles, and often the same user prompts across every turn of a conversation. Standard API gateways forward all of it to cloud LLMs regardless.

**Isartor sits between your tools and the cloud.** It intercepts every prompt and runs a cascade of local algorithms — from sub-millisecond hashing to in-process neural inference — to resolve requests before they reach the network. Only the genuinely hard prompts make it through.

The result: **lower costs, lower latency, and less data leaving your perimeter.**

| | Without Isartor | With Isartor |
|:--|:----------------|:-------------|
| Repeated prompts | Full cloud round-trip every time | Answered locally in < 1 ms |
| Similar prompts ("Price?" / "Cost?") | Full cloud round-trip every time | Matched semantically, answered locally in 1–5 ms |
| System instructions (CLAUDE.md, copilot-instructions) | Sent in full on every request | Deduplicated and compressed per session |
| Simple FAQ / data extraction | Routed to GPT-4 / Claude | Resolved by embedded SLM in 50–200 ms |
| Complex reasoning | Routed to cloud | Routed to cloud ✓ |

---

## The Deflection Stack

Every request passes through five layers. Only prompts that survive the full stack reach the cloud.

```
Request ──► L1a Exact Cache ──► L1b Semantic Cache ──► L2 SLM Router ──► L2.5 Context Optimiser ──► L3 Cloud
                 │ hit                │ hit                 │ simple             │ compressed               │
                 ▼                    ▼                     ▼                    ▼                          ▼
              Instant             Instant             Local Answer      Smaller Prompt             Cloud Answer
```

| Layer | What It Does | How | Latency |
|:------|:-------------|:----|:--------|
| **L1a** Exact Cache | Traps duplicate prompts and agent loops | `ahash` deterministic hashing | < 1 ms |
| **L1b** Semantic Cache | Catches paraphrases ("Price?" ≈ "Cost?") | Cosine similarity via pure-Rust `candle` embeddings | 1–5 ms |
| **L2** SLM Router | Resolves simple queries locally | Embedded Small Language Model (Qwen-1.5B via `candle` GGUF) | 50–200 ms |
| **L2.5** Context Optimiser | Compresses repeated instructions per session | Dedup + minify (CLAUDE.md, copilot-instructions) | < 1 ms |
| **L3** Cloud Logic | Routes complex prompts to OpenAI / Anthropic / Azure | Load balancing with retry and fallback | Network-bound |

### Benchmark results

| Workload | Deflection Rate | Detail |
|:---------|:---------------:|:-------|
| Warm agent session (Claude Code, 20 prompts) | **95%** | L1a 80% · L1b 10% · L2 5% · L3 5% |
| Repetitive FAQ loop (1,000 prompts) | **60%** | L1a 41% · L1b 19% · L3 40% |
| Diverse code-generation tasks (78 prompts) | **38%** | Exact-match duplicates only; all unique tasks route to L3 |

P50 latency for a cache hit: **0.3 ms.** [Full benchmark methodology →](benchmarks/README.md)

---

## AI Tool Integrations

One command connects your favourite tool. No proxy, no MITM, no CA certificates.

| Tool | Command | Mechanism |
|:-----|:--------|:----------|
| **GitHub Copilot CLI** | `isartor connect copilot` | MCP server (stdio or HTTP/SSE at `/mcp/`) |
| **GitHub Copilot in VS Code** | `isartor connect copilot-vscode` | Managed `settings.json` debug overrides |
| **OpenClaw** | `isartor connect openclaw` | Provider patch for `openclaw.json` |
| **Claude Code** | `isartor connect claude` | `ANTHROPIC_BASE_URL` override |
| **Claude Code + Copilot** | `isartor connect claude-copilot` | Claude base URL + Copilot-backed L3 |
| **Cursor IDE** | `isartor connect cursor` | Base URL + MCP registration at `/mcp/` |
| **OpenAI Codex CLI** | `isartor connect codex` | `OPENAI_BASE_URL` override |
| **Gemini CLI** | `isartor connect gemini` | `GEMINI_API_BASE_URL` override |
| **OpenCode** | `isartor connect opencode` | Global provider + auth config |
| **Any OpenAI-compatible tool** | `isartor connect generic` | Configurable env var override |

[Full integration guides →](https://isartor-ai.github.io/Isartor/integrations/overview.html)

---

## Drop-In for Any OpenAI SDK

Isartor is fully OpenAI-compatible and Anthropic-compatible. Point any existing SDK at it by changing one URL:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-isartor-api-key",
)

# First call → routed to cloud (L3), cached on return
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain the builder pattern in Rust"}],
)

# Second identical call → answered from L1a cache in < 1 ms
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain the builder pattern in Rust"}],
)
```

Works with the official Python/Node SDKs, LangChain, LlamaIndex, AutoGen, CrewAI, OpenClaw, or any OpenAI-compatible client.

---

## Scales from Laptop to Cluster

The same binary adapts from a developer laptop to a multi-replica Kubernetes deployment. Switch modes entirely through environment variables — no code changes, no recompilation.

| Component | Laptop (Single Binary) | Enterprise (K8s) |
|:----------|:-----------------------|:------------------|
| **L1a Cache** | In-memory LRU | Redis cluster (shared across replicas) |
| **L1b Embeddings** | In-process `candle` BertModel | External TEI sidecar |
| **L2 SLM** | Embedded `candle` GGUF inference | Remote vLLM / TGI (GPU pool) |
| **L2.5 Optimiser** | In-process | In-process |
| **L3 Cloud** | Direct to provider | Direct to provider |

```bash
# Flip to enterprise mode — just env vars, same binary
export ISARTOR__CACHE_BACKEND=redis
export ISARTOR__REDIS_URL=redis://redis-cluster.svc:6379
export ISARTOR__ROUTER_BACKEND=vllm
export ISARTOR__VLLM_URL=http://vllm.svc:8000
```

---

## Observability

Built-in **OpenTelemetry** traces and **Prometheus** metrics — no extra instrumentation.

- **Distributed traces** — root span `gateway_request` with child spans per layer (`l1a_exact_cache`, `l1b_semantic_cache`, `l2_classify_intent`, `context_optimise`, `l3_cloud_llm`).
- **Prometheus metrics** — `isartor_request_duration_seconds`, `isartor_layer_duration_seconds`, `isartor_requests_total`.
- **ROI tracking** — `isartor_tokens_saved_total` counts tokens that never left your infrastructure. Pipe it into Grafana to prove savings.

```bash
export ISARTOR__ENABLE_MONITORING=true
export ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317
```

[Observability guide →](https://isartor-ai.github.io/Isartor/observability/metrics-tracing.html)

---

## CLI Reference

```
isartor up                     Start the API gateway
isartor up --detach            Start in background
isartor up copilot             Start gateway + Copilot CONNECT proxy
isartor stop                   Stop a running instance
isartor demo                   Run the deflection demo (no API key needed)
isartor init                   Generate a commented config scaffold
isartor set-key -p openai      Configure your LLM provider API key
isartor stats                  Prompt totals, layer hits, routing history
isartor stats --by-tool        Per-tool cache hits, latency, errors
isartor update                 Self-update to the latest release
isartor connect <tool>         Connect an AI tool (see integrations above)
```

---

## Documentation

📚 **[isartor-ai.github.io/Isartor](https://isartor-ai.github.io/Isartor/)**

| | |
|:--|:--|
| [Getting Started](https://isartor-ai.github.io/Isartor/getting-started/installation.html) | Installation, first request, config basics |
| [Architecture](https://isartor-ai.github.io/Isartor/concepts/architecture.html) | Deflection Stack deep dive, trait provider pattern |
| [Integrations](https://isartor-ai.github.io/Isartor/integrations/overview.html) | Copilot, Cursor, Claude, Codex, Gemini, generic |
| [Deployment](https://isartor-ai.github.io/Isartor/deployment/level1-minimal.html) | Minimal → Sidecar → Enterprise (K8s) → Air-Gapped |
| [Configuration](https://isartor-ai.github.io/Isartor/configuration/reference.html) | Every environment variable and config key |
| [Observability](https://isartor-ai.github.io/Isartor/observability/metrics-tracing.html) | Spans, metrics, Grafana dashboards |
| [Performance Tuning](https://isartor-ai.github.io/Isartor/observability/performance-tuning.html) | Deflection measurement, SLO/SLA templates |
| [Troubleshooting](https://isartor-ai.github.io/Isartor/development/troubleshooting.html) | Common issues, diagnostics, FAQ |
| [Contributing](https://isartor-ai.github.io/Isartor/development/contributing.html) | Dev setup, PR guidelines |
| [Governance](GOVERNANCE.md) | Independence, license stability, decision-making |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and PR guidelines.

```bash
cargo build && cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

---

## License

Apache License, Version 2.0 — see [LICENSE](LICENSE).

Isartor is and will remain open source. No bait-and-switch relicensing. See [GOVERNANCE.md](GOVERNANCE.md) for the full commitment.

---

<p align="center">
  <sub>If Isartor saves you tokens, consider giving it a ⭐</sub>
</p>

# 🏛️ Isartor

<p align="center">
  <img src="docs/logo.png" alt="Isartor Logo" width="400">
</p>

**The Edge-Native AI Orchestration Gateway.**

[![CI Status](https://github.com/isartor-ai/Isartor/actions/workflows/ci.yml/badge.svg)](https://github.com/isartor-ai/Isartor/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/badge/Discord-Community-7289DA?logo=discord)](https://discord.gg/placeholder)

---

## The Elevator Pitch

Modern AI applications are drowning in API costs and network latency. Standard gateways act as dumb pipes, blindly forwarding every trivial "Hello" to heavyweight cloud models. **Isartor** fixes this at the edge. By embedding lightweight ML models (SLMs and sentence embedders) directly into a high-performance Rust binary, Isartor classifies and resolves simple requests in-process. This "Edge-Native Intelligence" approach slashes token burn by up to 80%, eliminates unnecessary network hops, and ensures sensitive data never leaves your infrastructure unless absolutely necessary.

## Key Features

### Pluggable Trait Provider Architecture

- **Minimalist Single-Binary Mode:**
  - Runs fully embedded SLMs (Gemma-2, Qwen2-1.5B) and fastembed semantic cache in-process.
  - Zero external dependencies: in-memory LRU cache, no Redis, no vLLM, no sidecars.
  - Ideal for edge devices, air-gapped environments, and rapid prototyping.

- **Enterprise K8s Mode:**
  - Switches to Redis for cache and vLLM/TGI for SLM routing via config (no code changes).
  - Horizontally scalable: stateless gateway pods, shared Redis cache, GPU inference pool.
  - Designed for multi-replica Kubernetes, managed observability, and high throughput.

- **Sub-Millisecond Semantic Cache:** In-process vector search for instant semantic matches.
- **Multi-Layer Funnel:** Sequential pipeline (Auth → Cache → SLM Triage → Cloud) short-circuits early to minimize cost and latency.
- **Observability-First:** OpenTelemetry, Jaeger, Tempo, Prometheus, Grafana.

## Architecture

![Architecture Diagram](docs/images/architecture_diagram.png)

Isartor uses a **Multi-Layer Funnel** approach to request orchestration. Every incoming prompt passes through a series of "short-circuit" layers. Layer 1 (Semantic Cache) uses embedded embeddings to find instant matches. Layer 2 (SLM Triage) classifies the intent; if the task is simple (e.g., "What time is it?"), it is resolved by an in-process Small Language Model. Only "Complex" intents that require reasoning or world-knowledge are forwarded to Layer 3 (Cloud LLMs).

## Quick Start

Run Isartor instantly using Docker:

```bash
docker run -p 8080:8080 \
  -e ISARTOR__GATEWAY_API_KEY="your-secret" \
  -e ISARTOR__LLM_PROVIDER="openai" \
  -e ISARTOR__EXTERNAL_LLM_API_KEY="sk-..." \
  ghcr.io/isartor-ai/isartor:latest
```

Test it with `curl`:

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "X-API-Key: your-secret" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Calculate 2+2"}'
```

## Local Development

### Prerequisites

- **Rust Toolchain**: [Install Rust](https://rustup.rs/) (Stable 1.75+)
- **CMake**: Required for building some ML backends.

### Build from Source

```bash
git clone https://github.com/isartor-ai/Isartor.git
cd Isartor
cargo build --release
```

The binary will be available at `./target/release/isartor`.

## Configuration

Isartor is configured via environment variables (prefixed with `ISARTOR__`) or a YAML/TOML configuration file.

```yaml
# isartor.yaml
host_port: "0.0.0.0:8080"
cache_mode: "both"
llm_provider: "azure"
external_llm_model: "gpt-4o-mini"
enable_monitoring: true
```

## Deployment Modes: Minimalist to Enterprise

## Deployment Modes: Minimalist Single-Binary vs Enterprise K8s

Isartor uses a **Pluggable Trait Provider** (Hexagonal Architecture) pattern. The same binary adapts to any deployment scale — from edge devices to multi-replica Kubernetes clusters — by selecting backends at startup via environment variables.

| Layer           | Minimalist Single-Binary           | Enterprise K8s                |
|:---------------:|:----------------------------------:|:-----------------------------:|
| **L1a Cache**   | In-memory LRU (ahash + parking_lot)| Redis cluster (shared cache)  |
| **L1b Semantic**| Fastembed CPU (in-process)         | External TEI (optional)       |
| **L2 Router**   | Embedded Candle/Qwen2 (in-process) | Remote vLLM/TGI server        |
| **L3 Fallback** | Cloud LLM (OpenAI/Anthropic)       | Cloud LLM (OpenAI/Anthropic)  |

**Switching Modes:**
Just set environment variables — no code changes or recompilation required.


```bash
# Switch cache to Redis
export ISARTOR__CACHE_BACKEND=redis
export ISARTOR__REDIS_URL=redis://redis-cluster.svc:6379

# Switch router to remote vLLM
export ISARTOR__ROUTER_BACKEND=vllm
export ISARTOR__VLLM_URL=http://vllm.svc:8000
export ISARTOR__VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
```

> For full architectural details see [`docs/2-ARCHITECTURE.md`](docs/2-ARCHITECTURE.md).

## License

Isartor is open-source software licensed under the **Apache License, Version 2.0**.

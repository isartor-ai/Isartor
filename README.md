# 🏛️ Isartor

**The Edge-Native AI Orchestration Gateway.**

[![CI Status](https://github.com/isartor-ai/Isartor/actions/workflows/ci.yml/badge.svg)](https://github.com/isartor-ai/Isartor/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/badge/Discord-Community-7289DA?logo=discord)](https://discord.gg/placeholder)

---

## The Elevator Pitch

Modern AI applications are drowning in API costs and network latency. Standard gateways act as dumb pipes, blindly forwarding every trivial "Hello" to heavyweight cloud models. **Isartor** fixes this at the edge. By embedding lightweight ML models (SLMs and sentence embedders) directly into a high-performance Rust binary, Isartor classifies and resolves simple requests in-process. This "Edge-Native Intelligence" approach slashes token burn by up to 80%, eliminates unnecessary network hops, and ensures sensitive data never leaves your infrastructure unless absolutely necessary.

## Key Features

- **🚀 Edge-Native Intelligence** – Runs SLMs (Gemma-2) and embedding models (BGE) embedded directly in the Rust binary via ONNX/Candle.
- **⚡ Sub-Millisecond Semantic Cache** – In-process vector search identifies semantically identical prompts instantly without external sidecars.
- **🛡️ Multi-Layer Funnel** – A sequential pipeline (Auth → Cache → SLM Triage → Cloud) that short-circuits as early as possible.
- **📦 Single-Binary Deployment** – Compiles to a tiny, static musl binary (~5MB) for distroless containers or bare-metal edge devices.
- **☸️ Kubernetes-Native** – First-class support for Helm, Prometheus, and horizontal scaling.
- **📊 Observability-First** – Full OpenTelemetry integration with distributed tracing (Jaeger/Tempo) and granular metrics (Grafana).

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

## License

Isartor is open-source software licensed under the **Apache License, Version 2.0**.

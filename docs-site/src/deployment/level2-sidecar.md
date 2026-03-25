# Level 2 — Sidecar Deployment

> **Split architecture: Isartor firewall + llama.cpp generation sidecar on a single host.**

This guide covers deploying Isartor with a dedicated AI sidecar for generation. The firewall delegates Layer 2 inference to a lightweight llama.cpp container via HTTP, while Layer 1 semantic cache embeddings run **in-process** via candle BertModel (no embedding sidecar required). The overall stack runs on a single machine via Docker Compose.

---

## When to Use Level 2

| ✅ Good Fit | ❌ Consider Level 1 or Level 3 |
| --- | --- |
| Single host with GPU (NVIDIA, AMD) | No GPU available → Level 1 embedded candle |
| Want GPU-accelerated Layer 2 generation | Multi-node scaling → Level 3 Kubernetes |
| Want full observability stack (Jaeger, Grafana) | Budget VPS (< 4 GB RAM) → Level 1 |
| Development with production-like topology | Auto-scaling inference pools → Level 3 |
| 10–100 concurrent users | > 100 concurrent users → Level 3 |

---

## Prerequisites

| Requirement | Minimum | Recommended |
| --- | --- | --- |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 10 GB | 20 GB (model cache) |
| **CPU** | 4 cores | 8+ cores |
| **GPU** (optional) | NVIDIA with 4 GB VRAM | NVIDIA with 8+ GB VRAM |
| **Docker** | 24.0+ | Latest |
| **Docker Compose** | v2.20+ | Latest |
| **NVIDIA Container Toolkit** (GPU) | Latest | Latest |

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Single Host                              │
│                                                                 │
│  ┌─────────────┐    ┌───────────────────┐    ┌──────────────┐  │
│  │   Client     │───▶│  Isartor Firewall │    │  Jaeger UI   │  │
│  │             │    │  :8080             │    │  :16686      │  │
│  └─────────────┘    │  (candle L1        │    └──────────────┘  │
│                     │   embeddings       │                      │
│                     │   built-in)        │                      │
│                     └──┬────────────────┘                       │
│                        │                                        │
│              HTTP :8081│                                        │
│                        ▼                                        │
│               ┌────────────┐                  ┌──────────────┐ │
│               │ slm-gen    │                  │  Grafana     │ │
│               │ Phi-3-mini │                  │  :3000       │ │
│               │ (llama.cpp)│                  └──────────────┘ │
│               └────────────┘                                    │
│                                               ┌──────────────┐ │
│               ┌─────────────────────────┐     │  Prometheus  │ │
│               │    OTel Collector :4317  │────▶│  :9090       │ │
│               └─────────────────────────┘     └──────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Optional: slm-embed :8082 (llama.cpp)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Services

| Service | Image | Port | Purpose | Memory Limit |
| --- | --- | --- | --- | --- |
| **gateway** | `isartor:latest` (built) | 8080 | Prompt Firewall (includes candle BertModel for Layer 1 embeddings) | 256 MB |
| **slm-generation** | `ghcr.io/ggml-org/llama.cpp:server` | 8081 | Phi-3-mini-4k (Q4_K_M) — intent classification + generation | 4 GB |
| **slm-embedding** *(optional)* | `ghcr.io/ggml-org/llama.cpp:server` | 8082 | all-MiniLM-L6-v2 (Q8_0) — external embedding sidecar (default uses in-process candle) | 512 MB |
| **otel-collector** | `otel/opentelemetry-collector-contrib:0.96.0` | 4317 | OTLP gRPC receiver | 128 MB |
| **jaeger** | `jaegertracing/all-in-one:1.55` | 16686 | Distributed tracing UI | 256 MB |
| **prometheus** | `prom/prometheus:v2.51.0` | 9090 | Metrics storage (7d retention) | 256 MB |
| **grafana** | `grafana/grafana:10.4.0` | 3000 | Dashboards | 256 MB |

---

## Quick Start (CPU Only)

### 1. Clone the Repository

```bash
git clone https://github.com/isartor-ai/isartor.git
cd isartor/docker
```

### 2. Configure Layer 3 (Optional)

Layers 0–2 work without a cloud LLM key. If you want Layer 3 fallback:

```bash
cp .env.full.example .env.full
```

Edit `.env.full` and set your provider:

```bash
ISARTOR__LLM_PROVIDER=openai
ISARTOR__EXTERNAL_LLM_MODEL=gpt-4o-mini
ISARTOR__EXTERNAL_LLM_API_KEY=sk-...
```

### 3. Start the Full Stack

```bash
docker compose -f docker-compose.sidecar.yml up --build
```

First launch downloads model files (~1.5 GB for Phi-3 + ~50 MB for MiniLM). Subsequent starts use the cached `isartor-slm-models` volume.

### 4. Wait for Health Checks

The firewall waits for both sidecars to become healthy before starting:

```bash
docker compose -f docker-compose.sidecar.yml ps
```

All services should show `healthy` or `running`.

### 5. Verify

```bash
# Health check
curl http://localhost:8080/healthz

# Test the firewall
curl -s http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?"}' | jq .

# If you enabled gateway auth, add:
#   -H "X-API-Key: your-secret-key"

# Check traces in Jaeger
open http://localhost:16686
```

---

## GPU Passthrough (NVIDIA)

To enable GPU acceleration for the llama.cpp sidecars:

### 1. Install NVIDIA Container Toolkit

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Add GPU Resources to Compose

Create a `docker-compose.gpu.override.yml`:

```yaml
services:
  slm-generation:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # The default --n-gpu-layers 99 in docker-compose.sidecar.yml
    # already offloads all layers to GPU when available.

  slm-embedding:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. Start with GPU Override

```bash
docker compose \
  -f docker-compose.sidecar.yml \
  -f docker-compose.gpu.override.yml \
  up --build
```

### Expected GPU Impact

| Metric | CPU Only (8-core) | GPU (RTX 3060 12 GB) |
| --- | --- | --- |
| Phi-3 classification | 500–2000 ms | 30–100 ms |
| Phi-3 generation (256 tokens) | 5–15 s | 0.5–2 s |
| MiniLM embedding | 20–50 ms | 5–10 ms |

---

## Available Compose Files

The `docker/` directory contains several Compose configurations for different use cases:

| File | Description | Provider |
| --- | --- | --- |
| `docker-compose.sidecar.yml` | **Recommended.** Full stack with llama.cpp sidecars + observability | Any (configurable) |
| `docker-compose.yml` | Legacy stack with Ollama (heavier) | OpenAI |
| `docker-compose.azure.yml` | Legacy stack with Ollama, pre-configured for Azure OpenAI | Azure |
| `docker-compose.observability.yml` | Observability-focused stack (Ollama + OTel + Jaeger + Grafana) | Azure |

> **We recommend `docker-compose.sidecar.yml`** for all new deployments. The llama.cpp sidecars are ~30 MB each vs. Ollama's ~1.5 GB.

---

## Environment Variables (Level 2 Specific)

These variables are relevant to the sidecar architecture. For the full reference, see the [Configuration Reference](../configuration/reference.md).

### Firewall ↔ Sidecar Communication

| Variable | Default | Description |
| --- | --- | --- |
| `ISARTOR__LAYER2__SIDECAR_URL` | `http://127.0.0.1:8081` | Generation sidecar URL (use Docker service name in Compose: `http://slm-generation:8081`) |
| `ISARTOR__LAYER2__MODEL_NAME` | `phi-3-mini` | Model name for OpenAI-compatible requests |
| `ISARTOR__LAYER2__TIMEOUT_SECONDS` | `30` | HTTP timeout for generation calls |
| `ISARTOR__EMBEDDING_SIDECAR__SIDECAR_URL` | `http://127.0.0.1:8082` | Embedding sidecar URL — optional (default uses in-process candle; use `http://slm-embedding:8082` in Compose) |
| `ISARTOR__EMBEDDING_SIDECAR__MODEL_NAME` | `all-minilm` | Embedding model name (sidecar only) |
| `ISARTOR__EMBEDDING_SIDECAR__TIMEOUT_SECONDS` | `10` | HTTP timeout for embedding calls (sidecar only) |

### Pluggable Backends

| Variable | Default | Description |
| --- | --- | --- |
| `ISARTOR__CACHE_BACKEND` | `memory` | In-process LRU — ideal for single-host Docker Compose |
| `ISARTOR__ROUTER_BACKEND` | `embedded` | In-process Candle SLM classification — no external dependency |

> **Scalability note:** These defaults are appropriate for Level 2 (single host). When moving to Level 3 (multi-replica K8s), switch to `cache_backend=redis` and `router_backend=vllm` for horizontal scaling.

### Cache

| Variable | Default | Description |
| --- | --- | --- |
| `ISARTOR__CACHE_MODE` | `both` | Use `both` — in-process candle BertModel provides semantic embeddings at all tiers |
| `ISARTOR__SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold for cache hits |

### Observability

| Variable | Default | Description |
| --- | --- | --- |
| `ISARTOR__ENABLE_MONITORING` | `true` (in Compose) | Enable OTel trace/metric export |
| `ISARTOR__OTEL_EXPORTER_ENDPOINT` | `http://otel-collector:4317` | OTel Collector gRPC endpoint |

---

## Operational Commands

### Logs

```bash
# All services
docker compose -f docker-compose.sidecar.yml logs -f

# Firewall only
docker compose -f docker-compose.sidecar.yml logs -f gateway

# Sidecars
docker compose -f docker-compose.sidecar.yml logs -f slm-generation slm-embedding
```

### Restart a Service

```bash
docker compose -f docker-compose.sidecar.yml restart gateway
```

### Tear Down (Preserve Model Cache)

```bash
docker compose -f docker-compose.sidecar.yml down
# Models persist in the 'isartor-slm-models' volume
```

### Tear Down (Clean Everything)

```bash
docker compose -f docker-compose.sidecar.yml down -v
# Removes all volumes including model cache — next start re-downloads models
```

### View Model Cache Size

```bash
docker volume inspect isartor-slm-models
```

---

## Networking Notes

- All services share a Docker bridge network created by Compose.
- The firewall references sidecars by Docker service name (`slm-generation`, `slm-embedding`), not `localhost`.
- Only the firewall (8080), Jaeger UI (16686), Grafana (3000), and Prometheus (9090) are exposed to the host.
- Sidecar ports (8081, 8082) are also exposed for debugging but can be removed in production by deleting the `ports:` mapping.

---

## Scaling Within Level 2

Before moving to Level 3, you can vertically scale Level 2:

| Optimisation | How |
| --- | --- |
| **More GPU VRAM** | Use larger quantisation (Q8_0 instead of Q4_K_M) for better quality |
| **Bigger model** | Swap Phi-3-mini for Phi-3-medium or Qwen2-7B in the Compose command |
| **More cache** | Increase `ISARTOR__CACHE_MAX_CAPACITY` and `ISARTOR__CACHE_TTL_SECS` |
| **Faster embedding** | Use `nomic-embed-text` (768-dim) for richer semantic matching |
| **More concurrency** | Scale horizontally with multiple firewall replicas behind a load balancer |

---

## Upgrading to Level 3

When a single host is no longer sufficient:

1. **Extract the firewall** into stateless Kubernetes pods (it's already stateless).
2. **Replace sidecars** with an auto-scaling inference pool (vLLM, TGI, or Triton).
3. **Add an internal load balancer** between firewall pods and the inference pool.
4. **Move observability** to a managed solution (Datadog, Grafana Cloud, Azure Monitor).

See [Level 3 — Enterprise Deployment](level3-enterprise.md) for the full Kubernetes guide.

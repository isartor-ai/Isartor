# Troubleshooting Guide

> **Common issues, diagnostic steps, and FAQ for operating Isartor.**

---

## Table of Contents

1. [Startup Errors](#1--startup-errors)
2. [Cache Issues](#2--cache-issues)
3. [Embedding & SLM Issues](#3--embedding--slm-issues)
4. [Cloud LLM Issues](#4--cloud-llm-issues)
5. [Observability Issues](#5--observability-issues)
6. [Performance & Degraded Operation](#6--performance--degraded-operation)
7. [Docker & Deployment Issues](#7--docker--deployment-issues)
8. [FAQ](#8--faq)

---

## 1 — Startup Errors

### 1.1 `Failed to initialize candle TextEmbedder`

**Symptom:** Firewall panics on startup with:

```text
Failed to initialize candle TextEmbedder (all-MiniLM-L6-v2)
```

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Model files not downloaded | Run once with internet access; candle auto-downloads to `~/.cache/huggingface/` |
| Corrupted model cache | Delete `~/.cache/huggingface/` and restart |
| Cache directory not writable (`Permission denied (os error 13)`) | Set `HF_HOME` (or `ISARTOR_HF_CACHE_DIR`) to a writable path (e.g. `/tmp/huggingface`). In Docker, mount a volume there: `-e HF_HOME=/tmp/huggingface -v isartor-hf:/tmp/huggingface`. |
| Insufficient memory | Ensure ≥ 256 MB available for the embedding model |

### 1.2 `Address already in use`

**Symptom:**

```text
Error: error creating server listener: Address already in use (os error 48)
```

**Fix:**

```bash
# Find the process using port 8080
lsof -i :8080
# Kill it, or change the port:
export ISARTOR__HOST_PORT=0.0.0.0:9090
```

### 1.3 `missing field` or config deserialization errors

**Symptom:**

```text
Error: missing field `layer2` in config
```

**Fix:** Ensure all required environment variables have the correct prefix
and separator. Isartor uses double-underscore (`__`) as separator:

```bash
# Correct:
export ISARTOR__LAYER2__SIDECAR_URL=http://127.0.0.1:8081

# Wrong:
export ISARTOR_LAYER2_SIDECAR_URL=http://127.0.0.1:8081
```

See the [Configuration Reference](5-CONFIGURATION-REF.md) for the full
list of variables.

### 1.4 `ISARTOR__GATEWAY_API_KEY` not set

**Symptom:** All requests return `401 Unauthorized`.

**Fix:**

```bash
export ISARTOR__GATEWAY_API_KEY=your-secret-key
```

The default is `changeme` — override it for any non-local deployment.

---

## 2 — Cache Issues

### 2.1 Low Cache Hit Rate

**Symptom:** Deflection rate below expected levels despite repeated traffic.

**Diagnostic steps:**

1. Check cache mode:
   ```bash
   echo $ISARTOR__CACHE_MODE   # should be "both" for most workloads
   ```

2. Check similarity threshold:
   ```bash
   echo $ISARTOR__SIMILARITY_THRESHOLD   # default: 0.85
   ```
   If too high (> 0.92), similar prompts won't match. Try lowering to 0.80.

3. Check TTL:
   ```bash
   echo $ISARTOR__CACHE_TTL_SECS   # default: 300
   ```
   Short TTL evicts entries before they can be reused.

4. Check Jaeger for `cosine_similarity` values on semantic cache spans.
   If scores are just below the threshold, lower it.

### 2.2 Stale Cache Responses

**Symptom:** Users receive outdated answers from cache.

**Fix:** Reduce TTL or restart the firewall to clear in-memory caches:

```bash
export ISARTOR__CACHE_TTL_SECS=60   # 1 minute
```

For Redis-backed caches, you can flush explicitly:

```bash
redis-cli -u $ISARTOR__REDIS_URL FLUSHDB
```

### 2.3 Redis Connection Refused

**Symptom:**

```text
Layer 1a: Redis connection error — falling through
```

**Diagnostic steps:**

1. Verify Redis is running:
   ```bash
   redis-cli -u $ISARTOR__REDIS_URL ping
   # Expected: PONG
   ```

2. Check network connectivity (especially in Docker/K8s):
   ```bash
   # Inside the firewall container:
   curl -v telnet://redis:6379
   ```

3. Verify the URL format:
   ```bash
   # Correct formats:
   export ISARTOR__REDIS_URL=redis://127.0.0.1:6379
   export ISARTOR__REDIS_URL=redis://user:password@redis.svc:6379/0
   ```

4. Check Redis memory limit — if Redis is OOM, it will reject writes.

**Fallback behaviour:** When Redis is unreachable, Isartor falls through
to the next layer. No data is lost, but deflection rate drops.

### 2.4 Cache Memory Growing Unbounded

**Symptom:** Firewall memory usage increases over time.

**Fix:** The in-memory cache uses bounded LRU eviction. Check:

```bash
echo $ISARTOR__CACHE_MAX_CAPACITY   # default: 10000
```

If set too high, reduce it. Each entry ≈ 2–4 KB, so 10K entries ≈ 20–40 MB.

---

## 3 — Embedding & SLM Issues

### 3.1 Slow Embedding Generation

**Symptom:** L1b latency > 10 ms.

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| CPU-bound contention | Increase CPU allocation for the container |
| Large prompt text | Embedder truncates to model max length (512 tokens), but longer text = more CPU |
| Cold start | First embedding call warms up the candle BertModel (~2 s). Subsequent calls are fast. |

### 3.2 SLM Sidecar Unreachable

**Symptom:**

```text
Layer 2: Failed to connect to SLM sidecar — falling through
```

**Diagnostic steps:**

1. Check if the sidecar is running:
   ```bash
   curl http://127.0.0.1:8081/v1/models
   ```

2. Verify configuration:
   ```bash
   echo $ISARTOR__LAYER2__SIDECAR_URL   # default: http://127.0.0.1:8081
   ```

3. Check the sidecar logs for errors (model loading, OOM, etc.).

4. Increase timeout if the sidecar is slow:
   ```bash
   export ISARTOR__LAYER2__TIMEOUT_SECONDS=60
   ```

**Fallback behaviour:** When the SLM sidecar is unreachable, Isartor
treats all requests as COMPLEX and forwards to Layer 3.

### 3.3 SLM Misclassification (SIMPLE ↔ COMPLEX)

**Symptom:** Users receive low-quality answers for complex questions
(misclassified as SIMPLE) or unnecessarily hit the cloud for simple ones.

**Diagnostic steps:**

1. In Jaeger, search for `router.decision` attribute to see classification
   distribution.

2. Send known-simple and known-complex prompts and check the classification:
   ```bash
   curl -s -X POST http://localhost:8080/api/chat \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" \
     -d '{"prompt": "What is 2+2?"}' | jq '.layer'
   # Expected: layer 2 (SIMPLE)
   ```

3. Consider switching to a larger SLM model for better classification accuracy.

### 3.4 Embedded Candle Engine Errors

**Symptom:**

```text
Layer 2: Embedded classification failed – falling through
```

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Model file missing | Set `ISARTOR__EMBEDDED__MODEL_PATH` to a valid GGUF file |
| Insufficient memory | Candle GGUF models need 1–4 GB RAM |
| Feature not compiled | Build with `--features embedded-inference` |

---

## 4 — Cloud LLM Issues

### 4.1 `502 Bad Gateway` from Layer 3

**Symptom:** Requests that reach Layer 3 return 502.

**Diagnostic steps:**

1. Check provider connectivity:
   ```bash
   curl -s $ISARTOR__EXTERNAL_LLM_URL \
     -H "Authorization: Bearer $ISARTOR__EXTERNAL_LLM_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}'
   ```

2. Verify API key is valid and has quota.

3. For Azure OpenAI, check deployment ID and API version:
   ```bash
   echo $ISARTOR__AZURE_DEPLOYMENT_ID
   echo $ISARTOR__AZURE_API_VERSION
   ```

### 4.2 Rate Limiting from Cloud Provider

**Symptom:** Intermittent 429 errors from the cloud LLM.

**Fix:**

- Increase deflection rate (lower threshold, longer TTL) to reduce cloud traffic.
- Request higher rate limits from your provider.
- Implement client-side retry with exponential backoff (application level).

### 4.3 Wrong Provider Configured

**Symptom:** Authentication errors or unexpected response formats.

**Fix:** Verify the provider matches the URL and API key:

```bash
# OpenAI
export ISARTOR__LLM_PROVIDER=openai
export ISARTOR__EXTERNAL_LLM_URL=https://api.openai.com/v1/chat/completions

# Azure
export ISARTOR__LLM_PROVIDER=azure
export ISARTOR__EXTERNAL_LLM_URL=https://YOUR-RESOURCE.openai.azure.com

# Anthropic
export ISARTOR__LLM_PROVIDER=anthropic
export ISARTOR__EXTERNAL_LLM_URL=https://api.anthropic.com/v1/messages

# xAI
export ISARTOR__LLM_PROVIDER=xai
export ISARTOR__EXTERNAL_LLM_URL=https://api.x.ai/v1/chat/completions

# Google Gemini
export ISARTOR__LLM_PROVIDER=gemini

# Mistral AI
export ISARTOR__LLM_PROVIDER=mistral

# Groq
export ISARTOR__LLM_PROVIDER=groq

# DeepSeek
export ISARTOR__LLM_PROVIDER=deepseek

# Cohere
export ISARTOR__LLM_PROVIDER=cohere

# Galadriel
export ISARTOR__LLM_PROVIDER=galadriel

# Hyperbolic
export ISARTOR__LLM_PROVIDER=hyperbolic

# HuggingFace Inference API
export ISARTOR__LLM_PROVIDER=huggingface

# Mira
export ISARTOR__LLM_PROVIDER=mira

# Moonshot (Kimi)
export ISARTOR__LLM_PROVIDER=moonshot

# Ollama (local — no API key required)
export ISARTOR__LLM_PROVIDER=ollama

# OpenRouter
export ISARTOR__LLM_PROVIDER=openrouter

# Perplexity
export ISARTOR__LLM_PROVIDER=perplexity

# Together AI
export ISARTOR__LLM_PROVIDER=together
```

---

## 5 — Observability Issues

### 5.1 No Traces in Jaeger

| Cause | Fix |
|-------|-----|
| Monitoring disabled | `export ISARTOR__ENABLE_MONITORING=true` |
| Wrong endpoint | `export ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317` |
| Collector not running | `docker compose -f docker-compose.observability.yml up otel-collector` |
| Firewall blocking gRPC | Ensure port 4317 is open between firewall and collector |

### 5.2 No Metrics in Prometheus

| Cause | Fix |
|-------|-----|
| Prometheus not scraping collector | Check `prometheus.yml` targets include `otel-collector:8889` |
| Collector metrics pipeline broken | Verify `otel-collector-config.yaml` exports to Prometheus |
| No requests sent yet | Send a test request — metrics appear after first request |

### 5.3 Grafana Shows "No Data"

| Cause | Fix |
|-------|-----|
| Data source not configured | Add Prometheus source: URL `http://prometheus:9090` |
| Wrong time range | Expand the time range in Grafana to cover the test period |
| Dashboard not provisioned | Check `docker/grafana/provisioning/` paths are mounted |

### 5.4 Console Shows "OTel disabled" Despite Setting env var

**Cause:** Config file takes precedence, or the env var prefix is wrong.

**Fix:**

```bash
# Correct (double underscore):
export ISARTOR__ENABLE_MONITORING=true

# Wrong (single underscore):
export ISARTOR_ENABLE_MONITORING=true  # ❌ not picked up
```

---

## 6 — Performance & Degraded Operation

### 6.1 High Tail Latency (P99 > 10 s)

**Diagnostic steps:**

1. Check which layer is the bottleneck:
   ```promql
   histogram_quantile(0.99,
     sum by (le, layer_name) (
       rate(isartor_layer_duration_seconds_bucket[5m])
     )
   )
   ```

2. Common causes:
   - L3 Cloud: provider is slow → switch to a faster model or provider.
   - L2 SLM: model inference is slow → use a smaller quantised model.
   - L1b Semantic: embedding is slow → check CPU contention.

### 6.2 Firewall OOM (Out of Memory)

**Diagnostic steps:**

1. Check cache capacity:
   ```bash
   echo $ISARTOR__CACHE_MAX_CAPACITY
   ```

2. Reduce capacity or switch to Redis backend.

3. If using embedded SLM, check model size vs. container memory limit.

### 6.3 Requests Queuing / High Connection Count

**Symptom:** Clients see connection timeouts or slow responses even for
cache hits.

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Too many concurrent requests | Scale horizontally (add replicas) |
| `spawn_blocking` pool exhaustion | Increase Tokio blocking threads: `TOKIO_WORKER_THREADS=8` |
| SLM inference blocking async runtime | Ensure SLM runs on blocking pool (default in Isartor) |

### 6.4 Degraded Mode (SLM Down, Cache Only)

When the SLM sidecar is unreachable, Isartor automatically degrades:

- L1a/L1b cache still works → cached requests are served.
- L2 SLM → all requests treated as COMPLEX → forwarded to L3.
- **Impact:** Higher cloud costs, but no downtime.

Monitor with:

```promql
# If SLM layer stops resolving requests, something is wrong
sum(rate(isartor_requests_total{final_layer="L2_SLM"}[5m])) == 0
```

---

## 7 — Docker & Deployment Issues

### 7.1 Docker Build Fails

**Symptom:** `cargo build` fails inside Docker.

**Common fixes:**

- Ensure Dockerfile uses the correct Rust toolchain version.
- For `aws-lc-rs` (TLS): install `cmake`, `gcc`, `make` in build stage.
- Check that `.dockerignore` isn't excluding required files.

### 7.2 Container Can't Reach Host Services

**Symptom:** Firewall inside Docker can't connect to sidecar on `localhost`.

**Fix:** Use Docker network names or `host.docker.internal`:

```bash
# docker-compose.yml
environment:
  - ISARTOR__LAYER2__SIDECAR_URL=http://sidecar:8081   # service name
  # or for host:
  - ISARTOR__LAYER2__SIDECAR_URL=http://host.docker.internal:8081
```

### 7.3 Health Check Failing

**Symptom:** Orchestrator keeps restarting the container.

**Fix:** The health endpoint is `GET /healthz`. Ensure the health check
matches:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
  interval: 10s
  timeout: 5s
  retries: 3
```

---

## 8 — FAQ

### Q: What is `cache_mode` and which should I use?

**A:** `cache_mode` controls which cache layers are active:

| Mode | What it does | Best for |
|------|-------------|----------|
| `exact` | Only SHA-256 hash match | Deterministic agent loops |
| `semantic` | Only cosine similarity | Diverse user queries |
| `both` | Exact first, then semantic | **Most workloads** (default) |

### Q: What happens if Redis goes down?

**A:** Isartor gracefully falls through. The exact cache layer logs a
warning and forwards the request downstream. No crash, no data loss.
Deflection rate drops until Redis recovers, and more requests reach the
cloud LLM (higher cost).

### Q: Can I change the embedding model?

**A:** Yes. The in-process embedder uses candle with a pure-Rust BertModel, which supports
multiple models. Set:

```bash
export ISARTOR__EMBEDDING_MODEL=bge-small-en-v1.5
```

The model is auto-downloaded on first startup. Note: changing the model
invalidates the semantic cache (different embedding dimensions/space).

### Q: How much does Isartor cost to run?

**A:** Isartor itself is free (Apache 2.0). The infrastructure cost depends
on your deployment:

| Mode | Estimated Cost |
|------|---------------|
| Minimalist (single binary, no GPU) | ~$5–15/month (small VM or container) |
| With SLM sidecar (CPU) | ~$20–50/month (4-core VM) |
| With SLM on GPU | ~$50–200/month (GPU instance) |
| Enterprise (K8s + Redis + vLLM) | ~$200–500/month |

The ROI comes from cloud LLM savings. At 70 % deflection and $0.01/1K
tokens, Isartor typically pays for itself within the first week.

### Q: Is Isartor production-ready?

**A:** Isartor is designed for production use with:

- ✅ Bounded, concurrent caches (no unbounded memory growth)
- ✅ Graceful degradation (every layer has a fallback)
- ✅ OpenTelemetry observability (traces, metrics, structured logs)
- ✅ Health check endpoint (`/healthz`)
- ✅ Configurable via environment variables (12-factor app)
- ✅ Integration tests covering all middleware layers

For enterprise deployments, use Redis-backed caches and a production
Kubernetes cluster. See the [Enterprise Guide](3-ENTERPRISE-GUIDE.md).

### Q: Can I use Isartor with LangChain / LlamaIndex / AutoGen?

**A:** Yes. Isartor exposes an OpenAI-compatible API. Point any SDK at
the firewall URL:

```python
import openai
client = openai.OpenAI(
    base_url="http://your-isartor-host:8080/v1",
    api_key="your-gateway-key",
)
```

See [Integrations](4-INTEGRATIONS.md) for full examples.

### Q: How do I upgrade Isartor?

**A:**

```bash
# Binary
cargo install --path . --force

# Docker
docker pull ghcr.io/isartor-ai/isartor:latest
docker compose up -d --pull always
```

In-memory caches are cleared on restart. Redis caches persist.

### Q: How do I monitor deflection rate in real-time?

**A:** Use the Grafana dashboard included in `dashboards/prometheus-grafana.json`
or the PromQL query:

```promql
1 - (
  sum(rate(isartor_requests_total{final_layer="L3_Cloud"}[5m]))
  /
  sum(rate(isartor_requests_total[5m]))
)
```

### Q: Can I run Isartor without any cloud LLM?

**A:** Partially. Layers 1 and 2 work standalone (cache + SLM). But
Layer 3 requires a cloud LLM API key. Without one, uncached COMPLEX
requests will return a 502 error. For fully local operation, ensure your
SLM can handle all traffic (set a very aggressive SIMPLE classification).

---

*← Back to [README](../README.md) · [Performance Tuning](PERFORMANCE-TUNING.md) · [Observability](6-OBSERVABILITY.md)*

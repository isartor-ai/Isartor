# Performance Tuning Guide

> **How to measure, tune, and operate Isartor for maximum deflection and
> minimum latency.**

---

## Table of Contents

1. [Understanding Deflection](#1--understanding-deflection)
2. [Measuring Deflection Rate](#2--measuring-deflection-rate)
3. [Tuning Configuration for Deflection](#3--tuning-configuration-for-deflection)
4. [Tuning Latency](#4--tuning-latency)
5. [Memory & Resource Tuning](#5--memory--resource-tuning)
6. [Cache Tuning Deep-Dive](#6--cache-tuning-deep-dive)
7. [SLM Router Tuning](#7--slm-router-tuning)
8. [Embedder Tuning](#8--embedder-tuning)
9. [SLO / SLA Goal Templates](#9--slo--sla-goal-templates)
10. [Scenario-Based Tuning Recipes](#10--scenario-based-tuning-recipes)
11. [PromQL Cheat Sheet](#11--promql-cheat-sheet)

---

## 1 — Understanding Deflection

**Deflection** = the percentage of requests resolved *before* Layer 3
(the external cloud LLM). A request is "deflected" if it is served by:

| Layer | Mechanism | Cost |
|-------|-----------|------|
| **L1a** — Exact Cache | SHA-256 hash match | $0 |
| **L1b** — Semantic Cache | Cosine similarity match | $0 |
| **L2** — SLM Triage | Local SLM classifies as SIMPLE and answers locally | $0 |

The **deflection rate** directly maps to cost savings. A 70 % deflection
rate means only 30 % of requests reach the paid cloud LLM.

---

## 2 — Measuring Deflection Rate

### 2.1 Via Prometheus / Grafana

The firewall emits `isartor_requests_total` with a `final_layer` label.
Use the following PromQL to compute the deflection rate:

```promql
# Overall deflection rate (last 1 hour)
1 - (
  sum(increase(isartor_requests_total{final_layer="L3_Cloud"}[1h]))
  /
  sum(increase(isartor_requests_total[1h]))
)
```

```promql
# Deflection rate by layer (pie chart)
sum by (final_layer) (rate(isartor_requests_total[5m]))
```

```promql
# Exact-cache deflection only
sum(increase(isartor_requests_total{final_layer="L1a_ExactCache"}[1h]))
/
sum(increase(isartor_requests_total[1h]))
```

### 2.2 Via the API

Send a test batch and count response `layer` values:

```bash
# Send 100 identical requests — expect 99 cache hits
for i in $(seq 1 100); do
  curl -s -X POST http://localhost:8080/api/chat \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $ISARTOR_API_KEY" \
    -d '{"prompt": "What is the capital of France?"}' \
  | jq '.layer'
done | sort | uniq -c
```

Expected output (ideal):

```text
   1 3       ← first request → cloud
  99 1       ← remaining → exact cache
```

### 2.3 Via Structured Logs

When `ISARTOR__ENABLE_MONITORING=true`, every request logs the final layer:

```bash
# grep JSON logs for final-layer distribution
cat logs.json | jq '.isartor.final_layer' | sort | uniq -c
```

### 2.4 Via Jaeger / Tempo

Filter traces by the `isartor.final_layer` tag:

| Goal | Search |
|------|--------|
| All cache hits | Tag `isartor.final_layer=L1a_ExactCache` or `L1b_SemanticCache` |
| SLM resolutions | Tag `isartor.final_layer=L2_SLM` |
| Cloud fallbacks | Tag `isartor.final_layer=L3_Cloud` |

---

## 3 — Tuning Configuration for Deflection

### 3.1 Cache Mode

| Variable | Values | Recommended |
|----------|--------|-------------|
| `ISARTOR__CACHE_MODE` | `exact`, `semantic`, `both` | **`both`** (default) |

- `exact` — Only identical prompts hit. Good for deterministic agent loops.
- `semantic` — Catches paraphrases ("Price?" ≈ "Cost?"). Higher hit rate but adds ~1–5 ms embedding cost.
- `both` — Exact check first (< 1 ms), then semantic if no exact hit. **Best of both worlds.**

### 3.2 Similarity Threshold

| Variable | Default | Range |
|----------|---------|-------|
| `ISARTOR__SIMILARITY_THRESHOLD` | `0.85` | `0.0`–`1.0` |

| Value | Effect |
|-------|--------|
| `0.95` | Very strict — only near-identical prompts match. Low false positives, lower hit rate. |
| `0.85` | Balanced — catches common paraphrases. **Recommended starting point.** |
| `0.75` | Aggressive — higher hit rate but risk of returning wrong cached answers. |
| `0.60` | Dangerous — high false-positive rate. Not recommended for production. |

**How to tune:**

1. Set `ISARTOR__ENABLE_MONITORING=true`.
2. Send representative traffic for 1 hour.
3. In Jaeger, search for `cosine_similarity` attribute on `l1b_semantic_cache_search` spans.
4. Plot the distribution. If most similarity scores cluster between 0.80–0.90, a threshold of 0.85 is good.
5. If you see many scores at 0.82–0.84 that *should* be hits, lower to 0.80.

### 3.3 Cache TTL

| Variable | Default | Description |
|----------|---------|-------------|
| `ISARTOR__CACHE_TTL_SECS` | `300` (5 min) | Time-to-live for cached responses |

- **Short TTL (60–120 s):** Good for rapidly changing data, real-time dashboards.
- **Medium TTL (300–600 s):** Balanced for most workloads.
- **Long TTL (1800+ s):** Maximises deflection for static Q&A / documentation bots.

### 3.4 Cache Capacity

| Variable | Default | Description |
|----------|---------|-------------|
| `ISARTOR__CACHE_MAX_CAPACITY` | `10000` | Max entries in each cache (LRU eviction) |

- Monitor eviction rate via `cache.evicted` span attribute on `l1b_semantic_cache_insert`.
- If eviction rate > 5 % of inserts, increase capacity or shorten TTL.
- Each cache entry ≈ 2–4 KB (prompt hash + response + optional 384-dim vector).

---

## 4 — Tuning Latency

### 4.1 Target Latencies by Layer

| Layer | Target (p95) | Typical Range |
|-------|-------------|---------------|
| L1a — Exact Cache | < 1 ms | 0.1–0.5 ms |
| L1b — Semantic Cache | < 10 ms | 1–5 ms |
| L2 — SLM Triage | < 300 ms | 50–200 ms (embedded), 100–500 ms (sidecar) |
| L3 — Cloud LLM | < 3 s | 500 ms – 5 s (network-bound) |

### 4.2 Measure with PromQL

```promql
# P95 latency by layer
histogram_quantile(0.95,
  sum by (le, layer_name) (
    rate(isartor_layer_duration_seconds_bucket[5m])
  )
)
```

```promql
# P95 end-to-end latency
histogram_quantile(0.95, rate(isartor_request_duration_seconds_bucket[5m]))
```

### 4.3 Reducing Latency

| Bottleneck | Symptom | Fix |
|-----------|---------|-----|
| Embedding | L1b > 10 ms | Use a lighter model or increase CPU allocation |
| SLM inference | L2 > 500 ms | Use quantised model (Q4_K_M GGUF), switch to embedded engine |
| Redis | L1a > 5 ms | Check network latency, use Redis cluster with read replicas |
| Cloud LLM | L3 > 5 s | Switch provider, use a smaller model, enable request timeout |

---

## 5 — Memory & Resource Tuning

### 5.1 Memory Budget

| Component | Memory Usage | Notes |
|-----------|-------------|-------|
| Exact cache (in-memory, 10K entries) | ~20–40 MB | Scales linearly with `cache_max_capacity` |
| Semantic cache (in-memory, 10K entries) | ~30–60 MB | 384-dim float32 vectors + response strings |
| candle embedder (all-MiniLM-L6-v2) | ~90 MB | Loaded at startup, constant |
| Candle GGUF model (embedded SLM) | ~1–4 GB | Depends on model quantisation |
| Tokio runtime | ~10–20 MB | Async task pool |
| **Total (minimalist mode)** | **~150–200 MB** | No embedded SLM |
| **Total (embedded mode)** | **~1.5–4.5 GB** | With embedded Candle SLM |

### 5.2 CPU Considerations

- Embedding generation runs on `spawn_blocking` (dedicated thread pool).
- Candle GGUF inference is CPU-bound; allocate ≥ 4 cores for embedded mode.
- The Tokio async runtime uses the default thread count (`num_cpus`).

### 5.3 Container Limits

```yaml
# docker-compose example
services:
  gateway:
    deploy:
      resources:
        limits:
          memory: 512M    # minimalist mode
          cpus: "2"
        # For embedded SLM mode:
        # limits:
        #   memory: 4G
        #   cpus: "4"
```

---

## 6 — Cache Tuning Deep-Dive

### 6.1 Exact vs. Semantic Cache Hit Analysis

```promql
# Exact cache hit rate
sum(rate(isartor_requests_total{final_layer="L1a_ExactCache"}[5m]))
/
sum(rate(isartor_requests_total[5m]))

# Semantic cache hit rate
sum(rate(isartor_requests_total{final_layer="L1b_SemanticCache"}[5m]))
/
sum(rate(isartor_requests_total[5m]))
```

### 6.2 Cache Backend: Memory vs. Redis

| Factor | In-Memory | Redis |
|--------|-----------|-------|
| Latency | ~0.1 ms | ~1–5 ms (network hop) |
| Capacity | Limited by process RAM | Limited by Redis memory |
| Multi-replica | ❌ No sharing | ✅ Shared across pods |
| Persistence | ❌ Lost on restart | ✅ Optional AOF/RDB |
| Recommended for | Single-instance, dev, edge | K8s, multi-replica, production |

Switch with:

```bash
export ISARTOR__CACHE_BACKEND=redis
export ISARTOR__REDIS_URL=redis://redis.svc:6379
```

### 6.3 When to Disable Semantic Cache

- Traffic is 100 % deterministic (exact same prompts repeated).
- Embedding overhead is unacceptable (< 1 ms budget).
- Set `ISARTOR__CACHE_MODE=exact`.

---

## 7 — SLM Router Tuning

### 7.1 Embedded vs. Sidecar

| Mode | Variable | Latency | Resource Usage |
|------|----------|---------|----------------|
| Embedded (Candle) | `ISARTOR__INFERENCE_ENGINE=embedded` | 50–200 ms | High CPU, 1–4 GB RAM |
| Sidecar (llama.cpp) | `ISARTOR__INFERENCE_ENGINE=sidecar` | 100–500 ms | Separate process, GPU optional |
| Remote (vLLM/TGI) | `ISARTOR__ROUTER_BACKEND=vllm` | 100–500 ms | Separate server, GPU recommended |

### 7.2 Model Selection

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| Phi-3-mini (Q4_K_M) | ~2 GB | Fast | Good |
| Gemma-2-2B-IT (Q4) | ~1.5 GB | Very fast | Good |
| Qwen-1.5-1.8B (Q4) | ~1.2 GB | Fastest | Adequate |
| Llama-3-8B (Q4) | ~4.5 GB | Slower | Best |

For intent classification (SIMPLE/COMPLEX), smaller models (1–3 B params)
are sufficient. Use the smallest model that meets your accuracy needs.

### 7.3 Tuning the Classification Prompt

The system prompt in `src/middleware/slm_triage.rs` determines classification
accuracy. If too many COMPLEX requests are misclassified as SIMPLE (bad
answers), consider:

1. Making the system prompt more specific to your domain.
2. Adding examples to the prompt (few-shot).
3. Switching to a larger model.

---

## 8 — Embedder Tuning

### 8.1 In-Process (candle)

The default embedder uses candle with `sentence-transformers/all-MiniLM-L6-v2` (pure-Rust BertModel):

- **384-dimensional** vectors
- **~90 MB** model footprint
- **1–5 ms** per embedding (CPU)
- Runs on `spawn_blocking` to avoid starving the Tokio runtime

### 8.2 Sidecar Embedder

For higher throughput or GPU acceleration:

```bash
export ISARTOR__EMBEDDING_SIDECAR__SIDECAR_URL=http://127.0.0.1:8082
export ISARTOR__EMBEDDING_SIDECAR__MODEL_NAME=all-minilm
export ISARTOR__EMBEDDING_SIDECAR__TIMEOUT_SECONDS=10
```

### 8.3 Embedding Model Selection

| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fastest | Good |
| bge-small-en-v1.5 | 384 | Fast | Better |
| bge-base-en-v1.5 | 768 | Moderate | Best |

Use 384-dim models for production. 768-dim models double memory usage
for marginal quality improvement in most use cases.

---

## 9 — SLO / SLA Goal Templates

### 9.1 Developer / Internal SLO

| Metric | Target | Measurement |
|--------|--------|-------------|
| Availability | 99.5 % | `up{job="isartor"}` over 30-day window |
| P95 latency (cache hit) | < 10 ms | `histogram_quantile(0.95, ...)` on L1 |
| P95 latency (end-to-end) | < 3 s | `histogram_quantile(0.95, ...)` on all |
| Deflection rate | > 50 % | `1 - (L3 / total)` over 24 h |
| Error rate | < 1 % | `rate(isartor_requests_total{http_status=~"5.."}[5m])` |

### 9.2 Production / Enterprise SLO

| Metric | Target | Measurement |
|--------|--------|-------------|
| Availability | 99.9 % | Multi-replica, health check monitoring |
| P95 latency (cache hit) | < 5 ms | Requires Redis or fast in-memory |
| P95 latency (end-to-end) | < 2 s | Optimised models, provider SLAs |
| P99 latency (end-to-end) | < 5 s | Tail latency budget |
| Deflection rate | > 70 % | Tuned thresholds + warm cache |
| Error rate | < 0.1 % | Circuit breakers, retries |
| Token savings | > 60 % | `isartor_tokens_saved_total` vs estimated total |

### 9.3 SLA Template (for downstream consumers)

```markdown
## Isartor Prompt Firewall SLA

**Availability:** 99.9 % monthly uptime (< 43.8 min downtime/month)
**Latency:** P95 end-to-end < 2 seconds
**Error Budget:** 0.1 % of requests may return 5xx
**Maintenance Window:** Sundays 02:00–04:00 UTC (excluded from SLA)

### Remediation
- Cache tier failure: automatic fallback to cloud LLM (degraded mode)
- SLM failure: automatic fallback to cloud LLM (degraded mode)
- Cloud LLM failure: 502 Bad Gateway returned, retry recommended

### Monitoring
- Health endpoint: GET /healthz
- Metrics endpoint: Prometheus scrape via OTel Collector on port 8889
- Dashboard: Grafana at http://<grafana-host>:3000
```

### 9.4 Alert Rules (Prometheus)

```yaml
groups:
  - name: isartor-slo
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(isartor_requests_total{http_status=~"5.."}[5m]))
          /
          sum(rate(isartor_requests_total[5m]))
          > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Isartor error rate exceeds 1%"

      - alert: HighP95Latency
        expr: |
          histogram_quantile(0.95, rate(isartor_request_duration_seconds_bucket[5m]))
          > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Isartor P95 latency exceeds 3 seconds"

      - alert: LowDeflectionRate
        expr: |
          1 - (
            sum(rate(isartor_requests_total{final_layer="L3_Cloud"}[1h]))
            /
            sum(rate(isartor_requests_total[1h]))
          ) < 0.5
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Isartor deflection rate below 50%"

      - alert: FirewallDown
        expr: up{job="isartor"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Isartor firewall is down"
```

---

## 10 — Scenario-Based Tuning Recipes

### Scenario A: Agentic Loop (High-Volume Identical Prompts)

**Profile:** Autonomous agent sends the same prompt hundreds of times per minute.

```bash
ISARTOR__CACHE_MODE=exact           # Semantic unnecessary for identical prompts
ISARTOR__CACHE_TTL_SECS=3600       # Long TTL — agent prompts are stable
ISARTOR__CACHE_MAX_CAPACITY=50000  # Large cache for many unique prompts
```

**Expected deflection:** 95–99 % (after warm-up).

### Scenario B: Customer Support Bot (Paraphrased Questions)

**Profile:** End users ask the same questions in different ways.

```bash
ISARTOR__CACHE_MODE=both
ISARTOR__SIMILARITY_THRESHOLD=0.80  # Lower threshold to catch paraphrases
ISARTOR__CACHE_TTL_SECS=1800       # 30 min — support answers change slowly
ISARTOR__CACHE_MAX_CAPACITY=10000
```

**Expected deflection:** 60–80 %.

### Scenario C: Code Generation (Low Cache Hit Rate)

**Profile:** Developers ask unique, complex coding questions.

```bash
ISARTOR__CACHE_MODE=both
ISARTOR__SIMILARITY_THRESHOLD=0.92  # High threshold — wrong cached code is costly
ISARTOR__CACHE_TTL_SECS=600        # Short TTL — code context changes quickly
ISARTOR__INFERENCE_ENGINE=embedded   # Let SLM handle simple code questions
```

**Expected deflection:** 20–40 % (SLM handles simple extraction).

### Scenario D: RAG Pipeline (Document Q&A)

**Profile:** Queries against a knowledge base; similar questions are common.

```bash
ISARTOR__CACHE_MODE=both
ISARTOR__SIMILARITY_THRESHOLD=0.83  # Moderate threshold
ISARTOR__CACHE_TTL_SECS=3600       # Documents change infrequently
ISARTOR__CACHE_MAX_CAPACITY=20000  # Large cache for document variation
```

**Expected deflection:** 50–70 %.

### Scenario E: Multi-Replica Kubernetes

**Profile:** Horizontally scaled behind a load balancer.

```bash
ISARTOR__CACHE_BACKEND=redis
ISARTOR__REDIS_URL=redis://redis-cluster.svc:6379
ISARTOR__ROUTER_BACKEND=vllm
ISARTOR__VLLM_URL=http://vllm.svc:8000
ISARTOR__VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
ISARTOR__CACHE_MODE=both
ISARTOR__SIMILARITY_THRESHOLD=0.85
```

**Benefit:** All replicas share the same cache → deflection rate applies
cluster-wide.

---

## 11 — PromQL Cheat Sheet

| What | Query |
|------|-------|
| Deflection rate (1 h) | `1 - (sum(increase(isartor_requests_total{final_layer="L3_Cloud"}[1h])) / sum(increase(isartor_requests_total[1h])))` |
| Request rate | `rate(isartor_requests_total[5m])` |
| Request rate by layer | `sum by (final_layer) (rate(isartor_requests_total[5m]))` |
| P50 latency | `histogram_quantile(0.50, rate(isartor_request_duration_seconds_bucket[5m]))` |
| P95 latency | `histogram_quantile(0.95, rate(isartor_request_duration_seconds_bucket[5m]))` |
| P99 latency | `histogram_quantile(0.99, rate(isartor_request_duration_seconds_bucket[5m]))` |
| Per-layer P95 | `histogram_quantile(0.95, sum by (le, layer_name) (rate(isartor_layer_duration_seconds_bucket[5m])))` |
| Tokens saved (daily) | `sum(increase(isartor_tokens_saved_total[24h]))` |
| Tokens saved by layer | `sum by (final_layer) (rate(isartor_tokens_saved_total[5m]))` |
| Est. daily cost savings ($0.01/1K tok) | `sum(increase(isartor_tokens_saved_total[24h])) / 1000 * 0.01` |
| Error rate | `sum(rate(isartor_requests_total{http_status=~"5.."}[5m])) / sum(rate(isartor_requests_total[5m]))` |
| Cache hit ratio (exact) | `sum(rate(isartor_requests_total{final_layer="L1a_ExactCache"}[5m])) / sum(rate(isartor_requests_total[5m]))` |
| Cache hit ratio (semantic) | `sum(rate(isartor_requests_total{final_layer="L1b_SemanticCache"}[5m])) / sum(rate(isartor_requests_total[5m]))` |

---

*← Back to [README](../README.md) · [Observability](6-OBSERVABILITY.md) · [Troubleshooting](TROUBLESHOOTING.md)*

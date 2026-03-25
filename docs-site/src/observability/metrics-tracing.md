# Metrics & Tracing

> **Definitive reference for Isartor's OpenTelemetry traces, metrics, structured logging, and observability stack — from local development to Kubernetes.**

---

## Overview

Isartor uses [OpenTelemetry](https://opentelemetry.io/) for distributed
tracing and metrics, plus `tracing-subscriber` with a JSON layer for
structured logging.

| Signal   | Protocol        | Default Endpoint           |
|----------|-----------------|----------------------------|
| Traces   | OTLP gRPC       | `http://localhost:4317`    |
| Metrics  | OTLP gRPC       | `http://localhost:4317`    |
| Logs     | stdout (JSON)   | —                          |

When `ISARTOR__ENABLE_MONITORING=false` (default), only the console log
layer is active — zero OTel overhead.

### Architecture

```text
┌─────────────┐                  ┌──────────────────┐
│  Isartor    │  OTLP gRPC      │  OTel Collector   │
│  Gateway    │─────────────────▶│  :4317            │
│             │  (traces +       │                   │
│             │   metrics)       │  Pipelines:       │
└─────────────┘                  │  traces → Jaeger  │
                                 │  metrics → Prom   │
                                 └───┬──────────┬────┘
                                     │          │
                          ┌──────────▼──┐  ┌────▼──────────┐
                          │   Jaeger    │  │  Prometheus   │
                          │   :16686    │  │  :9090        │
                          │   (UI)      │  │  (scrape)     │
                          └─────────────┘  └───────┬───────┘
                                                   │
                                           ┌───────▼───────┐
                                           │   Grafana     │
                                           │   :3000       │
                                           │  (dashboards) │
                                           └───────────────┘
```

---

## Enabling Monitoring

```bash
ISARTOR__ENABLE_MONITORING=true
ISARTOR__OTEL_EXPORTER_ENDPOINT=http://localhost:4317
RUST_LOG=info,h2=warn,hyper=warn,tower=warn       # optional override
```

When `ISARTOR__ENABLE_MONITORING=false` (the default), Isartor uses console-only logging via `tracing-subscriber` with `RUST_LOG` filtering. No OTel SDK is initialised — zero overhead.

---

## Telemetry Initialisation (`src/telemetry.rs`)

`init_telemetry()` returns an **`OtelGuard`** (RAII). The guard holds the
`SdkTracerProvider` and `SdkMeterProvider`; dropping it flushes pending
telemetry and shuts down exporters gracefully.

| Component              | Description                                    |
|------------------------|------------------------------------------------|
| **JSON stdout layer**  | Structured logs emitted as JSON when monitoring is on |
| **Pretty console layer** | Human-readable output when monitoring is off  |
| **OTLP trace exporter** | gRPC via `opentelemetry-otlp` → Collector     |
| **OTLP metric exporter** | gRPC via `opentelemetry-otlp` → Collector    |
| **EnvFilter**          | Reads `RUST_LOG`, defaults to `info,h2=warn,hyper=warn,tower=warn` |

Service identity:

```text
service.name    = "isartor-gateway"
service.version = env!("CARGO_PKG_VERSION")   # e.g. "0.1.0"
```

---

## Distributed Traces — Span Reference

Every request gets a **root span** (`gateway_request`) from the monitoring
middleware. Child spans are created per-layer:

### Root Span

| Span Name          | Source                            | Key Attributes                                                               |
|--------------------|-----------------------------------|------------------------------------------------------------------------------|
| `gateway_request`  | `src/middleware/monitoring.rs`     | `http.method`, `http.route`, `http.status_code`, `client.address`, `isartor.final_layer` |

`http.status_code` and `isartor.final_layer` are recorded **after** the
response returns (empty → filled pattern).

### Layer 0 — Auth

| Span Name | Source | Key Attributes |
|-----------|--------|----------------|
| *(inline `tracing::debug!`/`warn!`)* | `src/middleware/auth.rs` | — |

Auth is lightweight; no dedicated span is created. Events are logged at
debug/warn level.

### Layer 1a — Exact Cache

| Span Name              | Source                     | Key Attributes                                     |
|------------------------|----------------------------|----------------------------------------------------|
| `l1a_exact_cache_get`  | `src/adapters/cache.rs`    | `cache.backend` (`memory`\|`redis`), `cache.key`, `cache.hit` |
| `l1a_exact_cache_put`  | `src/adapters/cache.rs`    | `cache.backend`, `cache.key`, `response_len`       |

### Layer 1b — Semantic Cache

| Span Name                     | Source                    | Key Attributes                                                         |
|-------------------------------|---------------------------|------------------------------------------------------------------------|
| `l1b_semantic_cache_search`   | `src/vector_cache.rs`     | `cache.entries_scanned`, `cache.hit`, **`cosine_similarity`**          |
| `l1b_semantic_cache_insert`   | `src/vector_cache.rs`     | `cache.evicted`, `cache.size_after`                                    |

> **`cosine_similarity`** — the best-match score formatted to 4 decimal
> places. **This is the key attribute for tuning the similarity threshold.**

### Layer 2 — SLM Triage

| Span Name              | Source                            | Key Attributes                                                           |
|------------------------|-----------------------------------|--------------------------------------------------------------------------|
| `layer2_slm`           | `src/middleware/slm_triage.rs`    | `slm.complexity_score` (`TEMPLATE`\|`SNIPPET`\|`COMPLEX`; legacy binary mode: `SIMPLE`\|`COMPLEX`) |
| `l2_classify_intent`   | `src/adapters/router.rs`          | `router.backend` (`embedded_candle`\|`remote_vllm`), **`router.decision`**, `router.model`, `router.url`, `prompt_len` |

### Layer 2.5 — Context Optimiser

| Span Name                       | Source                                | Key Attributes                                                       |
|---------------------------------|---------------------------------------|----------------------------------------------------------------------|
| `layer2_5_context_optimizer`    | `src/middleware/context_optimizer.rs`  | `context.bytes_saved`, `context.strategy` (e.g. `"classifier+dedup"`, `"classifier+log_crunch"`) |

When L2.5 modifies the request body, it also sets the response header `x-isartor-context-optimized: bytes_saved=<N>`.

### Layer 3 — Cloud LLM

| Span Name      | Source              | Key Attributes                                           |
|----------------|---------------------|----------------------------------------------------------|
| `layer3_llm`   | `src/handler.rs`    | `ai.prompt.length_bytes`, **`provider.name`**, **`model`** |

---

## Custom Span Attributes — Quick Reference

These are the **Isartor-specific** attributes (beyond standard OTel
semantic conventions) that appear on spans and are useful for filtering
in Jaeger / Tempo:

| Attribute              | Type      | Where Set                   | Purpose                              |
|------------------------|-----------|-----------------------------|--------------------------------------|
| `isartor.final_layer`  | string    | Root `gateway_request` span | Which layer resolved the request     |
| `cache.hit`            | bool      | L1a and L1b spans           | Whether the cache lookup succeeded   |
| `cosine_similarity`    | string    | L1b search span             | Best cosine-similarity score (4 d.p) |
| `cache.entries_scanned`| u64       | L1b search span             | Entries scanned during similarity search |
| `cache.backend`        | string    | L1a get/put spans           | `"memory"` or `"redis"`              |
| `router.decision`      | string    | L2 classify span            | `"TEMPLATE"`, `"SNIPPET"`, or `"COMPLEX"` (tiered mode); `"SIMPLE"` or `"COMPLEX"` (binary mode) |
| `router.backend`       | string    | L2 classify span            | `"embedded_candle"` or `"remote_vllm"` |
| `context.bytes_saved`  | u64       | L2.5 optimizer span         | Bytes removed by compression pipeline  |
| `context.strategy`     | string    | L2.5 optimizer span         | Pipeline stages that modified content (e.g. `"classifier+dedup"`) |
| `provider.name`        | string    | L3 handler span             | e.g. `"openai"`, `"xai"`, `"azure"` |
| `model`                | string    | L3 handler span             | e.g. `"gpt-4o"`, `"grok-beta"`      |
| `http.status_code`     | u16       | Root span                   | HTTP response status code            |
| `client.address`       | string    | Root span                   | Client IP (from `x-forwarded-for`)   |

---

## OTel Metrics (`src/metrics.rs`)

Four instruments are registered as a singleton `GatewayMetrics` via `OnceLock`:

| Metric Name                          | Type       | Attributes                      | Description                              |
|--------------------------------------|------------|---------------------------------|------------------------------------------|
| `isartor_requests_total`             | Counter    | `final_layer`, `status_code`, `traffic_surface`, `client`, `endpoint_family`, `tool` | Total prompts processed |
| `isartor_request_duration_seconds`   | Histogram  | `final_layer`, `status_code`, `traffic_surface`, `client`, `endpoint_family` | End-to-end request duration |
| `isartor_layer_duration_seconds`     | Histogram  | `layer_name`, `tool`            | Per-layer latency                        |
| `isartor_tokens_saved_total`         | Counter    | `final_layer`, `traffic_surface`, `client`, `endpoint_family`, `tool` | Estimated tokens saved by early resolve |
| `isartor_errors_total`               | Counter    | `layer`, `error_class`, `tool`  | Error occurrences by layer / agent       |
| `isartor_retries_total`              | Counter    | `operation`, `attempts`, `outcome`, `tool` | Retry outcomes by agent |
| `isartor_cache_events_total`         | Counter    | `cache_layer`, `outcome`, `tool` | L1 / L1a / L1b hit-miss safety by agent |

### Where Metrics Are Recorded

| Call Site                        | Metrics Recorded                                           |
|----------------------------------|------------------------------------------------------------|
| `root_monitoring_middleware`      | `record_request_with_context()`, `record_tokens_saved_with_context()` (if early) |
| `proxy::connect::emit_proxy_decision()` | `record_request_with_context()`, `record_tokens_saved_with_context()` (if early) |
| `cache_middleware` (L1 hit)       | `record_layer_duration("L1a_ExactCache" \| "L1b_SemanticCache")` |
| `slm_triage_middleware` (L2 hit)  | `record_layer_duration("L2_SLM")`                         |
| `context_optimizer_middleware`    | `record_layer_duration("L2_5_ContextOptimiser")` (when bytes saved > 0) |
| `chat_handler` (L3)              | `record_layer_duration("L3_Cloud")`                        |

### Request Dimensions

Unified prompt telemetry distinguishes:

- `traffic_surface`: `gateway` or `proxy`
- `client`: `direct`, `openai`, `anthropic`, `copilot`, `claude`, `antigravity`, etc.
- `endpoint_family`: `native`, `openai`, or `anthropic`

### Token Estimation

`estimate_tokens(prompt)` uses the heuristic: `max(1, prompt.len() / 4)`.
This is intentionally conservative — the metric tracks **relative savings**
rather than precise token counts.

---

## ROI — `isartor_tokens_saved_total`

This is the **headline business metric**. Every request resolved before
Layer 3 (exact cache, semantic cache, or local SLM) avoids a round-trip
to the external LLM provider.

```promql
# Daily token savings
sum(increase(isartor_tokens_saved_total[24h]))

# Savings by layer
sum by (final_layer) (rate(isartor_tokens_saved_total[1h]))

# Prompt volume by traffic surface
sum by (traffic_surface) (rate(isartor_requests_total[5m]))

# Prompt volume by client
sum by (client) (rate(isartor_requests_total[5m]))

# Estimated cost savings (assuming $0.01 per 1K tokens)
sum(increase(isartor_tokens_saved_total[24h])) / 1000 * 0.01
```

Use this metric to justify infrastructure spend for the caching / SLM
layers.

---

## Docker Compose — Local Observability Stack

Use the provided compose file for local development:

```bash
cd docker
docker compose -f docker-compose.observability.yml up -d
```

| Service            | Port   | Purpose                          |
|--------------------|--------|----------------------------------|
| **OTel Collector** | 4317   | OTLP gRPC receiver               |
| **Jaeger**         | 16686  | Trace UI                         |
| **Prometheus**     | 9090   | Metrics scrape + query           |
| **Grafana**        | 3000   | Dashboards (anonymous admin)     |

Configuration files:

| File                              | Purpose                   |
|-----------------------------------|---------------------------|
| `docker/otel-collector-config.yaml` | Collector pipelines     |
| `docker/prometheus.yml`            | Scrape targets           |

### Pipeline Flow

```text
Isartor  ──OTLP gRPC──▶  OTel Collector ──▶  Jaeger    (traces)
                                          └──▶  Prometheus (metrics)
                                                     │
                                                     ▼
                                                  Grafana
```

### OTel Collector Configuration

The collector config is at `docker/otel-collector-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
      http:

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
  otlp:
    endpoint: "jaeger:4317"
    tls:
      insecure: true
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlp, debug]
    metrics:
      receivers: [otlp]
      exporters: [prometheus, debug]
```

### Prometheus Configuration

The Prometheus config is at `docker/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'otel-collector'
    scrape_interval: 5s
    static_configs:
      - targets: ['otel-collector:8889']
```

Prometheus scrapes the OTel Collector's Prometheus exporter on port 8889 every 5 seconds.

---

## Per-Tier Setup

### Level 1 — Minimal (Console Logs Only)

No observability stack is needed. Use `RUST_LOG` for structured console output:

```bash
ISARTOR__ENABLE_MONITORING=false
RUST_LOG=isartor=info
```

For debug-level output during development:

```bash
RUST_LOG=isartor=debug,tower_http=trace
```

### Level 2 — Docker Compose (Full Stack)

The `docker-compose.sidecar.yml` includes the complete observability stack:

```bash
cd docker
docker compose -f docker-compose.sidecar.yml up --build
```

Services included:

| Service | URL | Purpose |
|---------|-----|---------|
| **OTel Collector** | `localhost:4317` (gRPC) | Receives OTLP from gateway |
| **Jaeger UI** | `http://localhost:16686` | View distributed traces |
| **Prometheus** | `http://localhost:9090` | Query metrics |
| **Grafana** | `http://localhost:3000` | Dashboards (anonymous admin access) |

The gateway is pre-configured with:

```bash
ISARTOR__ENABLE_MONITORING=true
ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317
```

### Level 3 — Kubernetes (Managed or Self-Hosted)

| Approach | Recommended Stack | Notes |
|----------|-------------------|-------|
| **Self-managed** | OTel Collector DaemonSet + Jaeger Operator + kube-prometheus-stack | Full control, higher ops burden |
| **AWS** | AWS X-Ray + CloudWatch + Managed Grafana | ADOT Collector as sidecar/DaemonSet |
| **GCP** | Cloud Trace + Cloud Monitoring + Cloud Logging | Use OTLP exporter to Cloud Trace |
| **Azure** | Application Insights + Azure Monitor | Use Azure Monitor OpenTelemetry exporter |
| **Grafana Cloud** | Grafana Alloy + Grafana Cloud | Low ops, managed Prometheus + Tempo |
| **Datadog** | Datadog Agent + OTel Collector | Enterprise APM |

For all options, point the gateway at the collector:

```bash
ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector.isartor:4317
```

---

## Grafana Dashboard Queries (PromQL)

| Panel                 | PromQL                                                                                         |
|-----------------------|-----------------------------------------------------------------------------------------------|
| Request Rate          | `rate(isartor_requests_total[5m])`                                                            |
| P95 Latency           | `histogram_quantile(0.95, rate(isartor_request_duration_seconds_bucket[5m]))`                 |
| Layer Resolution      | `sum by (final_layer) (rate(isartor_requests_total[5m]))`                                     |
| Traffic Surface Split | `sum by (traffic_surface) (rate(isartor_requests_total[5m]))`                                 |
| Client Split          | `sum by (client) (rate(isartor_requests_total[5m]))`                                          |
| Per-Layer Latency     | `histogram_quantile(0.95, sum by (le, layer_name) (rate(isartor_layer_duration_seconds_bucket[5m])))` |
| Tokens Saved / Hour   | `sum(increase(isartor_tokens_saved_total[1h]))`                                               |
| Tokens Saved by Layer | `sum by (final_layer) (rate(isartor_tokens_saved_total[5m]))`                                 |
| Cache Hit Rate        | `rate(isartor_requests_total{final_layer=~"L1.*"}[5m]) / rate(isartor_requests_total[5m])`    |

---

## Jaeger — Useful Searches

| Goal                           | Search                                          |
|--------------------------------|------------------------------------------------|
| Slow requests (> 500 ms)      | Service `isartor-gateway`, Min Duration `500ms` |
| Cache misses                   | Tag `cache.hit=false`                           |
| Semantic cache tuning          | Tag `cosine_similarity` — sort by value         |
| Layer 3 fallbacks              | Tag `isartor.final_layer=L3_Cloud`              |
| SLM local resolutions          | Tag `router.decision=TEMPLATE` or `router.decision=SNIPPET` (tiered); `router.decision=SIMPLE` (binary) |

### Trace Anatomy

A typical trace for a cache-miss, locally-resolved request:

```text
isartor-gateway
  └─ HTTP POST /api/chat                       [250ms]
       ├─ Layer0_AuthCheck                       [0.1ms]
       ├─ Layer1_SemanticCache (MISS)            [5ms]
       ├─ Layer2_IntentClassifier                [80ms]
       │     intent=TEMPLATE, confidence=0.97
       └─ Layer2_LocalExecutor                   [160ms]
             model=phi-3-mini, tokens=42
```

---

## Built-in User Views

For quick operator checks without a separate telemetry stack:

```bash
isartor stats --gateway-url http://localhost:8080
isartor stats --gateway-url http://localhost:8080 --by-tool
```

Add `--gateway-api-key <key>` only when gateway auth is enabled.

`--by-tool` prints richer per-agent stats: requests, cache hits/misses,
average latency, retry count, error count, and L1a/L1b safety ratios.

Built-in JSON endpoints:

- `GET /health`
- `GET /debug/proxy/recent`
- `GET /debug/stats/prompts`
- `GET /debug/stats/agents`

---

## Alerting Rules

### Prometheus Alerting Rules

Create `docker/prometheus-alerts.yml`:

```yaml
groups:
  - name: isartor
    rules:
      - alert: HighErrorRate
        expr: rate(isartor_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Isartor error rate > 5% for 5 minutes"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(isartor_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Isartor P95 latency > 2s for 5 minutes"

      - alert: LowCacheHitRate
        expr: >
          rate(isartor_requests_total{final_layer=~"L1.*"}[15m]) /
          rate(isartor_requests_total[15m]) < 0.3
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "Cache hit rate below 30% — consider tuning similarity threshold"

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
          summary: "Isartor gateway is down"
```

---

## Troubleshooting

| Symptom                            | Cause                             | Fix                                         |
|------------------------------------|-----------------------------------|---------------------------------------------|
| No traces in Jaeger                | Monitoring disabled               | Set `ISARTOR__ENABLE_MONITORING=true`       |
| No traces in Jaeger                | Collector unreachable             | Verify `OTEL_EXPORTER_ENDPOINT` + port 4317 |
| No metrics in Prometheus           | Prometheus can't scrape collector | Check `prometheus.yml` targets              |
| Grafana "No data"                  | Data source misconfigured         | URL should be `http://prometheus:9090`      |
| Console shows "OTel disabled"      | Config precedence                 | Check env vars override file config         |
| `isartor_layer_duration_seconds` empty | No requests yet               | Send a test request                         |

---

*See also: [Configuration Reference](../configuration/reference.md) · [Performance Tuning](performance-tuning.md) · [Troubleshooting](../development/troubleshooting.md)*

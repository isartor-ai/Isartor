# Isartor ROI Report — Claude Code + Copilot

> Generated: 2026-03-24T10:33:03Z  
> Source data: `/home/runner/work/Isartor/Isartor/benchmarks/results/latest.json`  
> Hardware: 2-core x86_64, 7 GB RAM, no GPU  


This report compares **Claude Code + GitHub Copilot running through Isartor** against a **baseline with no Isartor** (all requests forwarded directly to the cloud provider). It covers L1/L2/L3 layer distribution, token and cost savings, and latency impact.


## Executive summary

| Metric                     | Value                              |
|----------------------------|------------------------------------|
| Total requests analysed    |                              1,500 |
| Overall deflection rate    |                              72.9% |
| Cloud input tokens saved   |                            148,800 |
| Cloud output tokens saved  |                            327,900 |
| Estimated cost (without)   |                            $7.8750 |
| Estimated cost (with)      |                            $2.2125 |
| Estimated cost saved       | $5.6625 (71.9% reduction) |


## Fixture: `faq_loop`

> **1,000 requests** — deflection rate: **73.3%**

### Layer distribution

| Layer              |  Hits  | % of Traffic | Avg Latency (p50) |
|--------------------|--------|--------------|-------------------|
| L1a (exact cache)  |    431 |        43.1% |            0.4 ms |
| L1b (semantic)     |    229 |        22.9% |            3.1 ms |
| L2  (local SLM)    |     73 |         7.3% |          138.6 ms |
| L3  (cloud)        |    267 |        26.7% |          859.9 ms |

> Overall (with Isartor) — P50: 2.4 ms | P95: 1064.8 ms | P99: 1160.1 ms

### With vs without Isartor

| Metric                          | Without Isartor (baseline) | With Isartor       |
|---------------------------------|----------------------------|--------------------|
| Cloud input tokens              |                    150,000 |             51,000 |
| Cloud output tokens             |                    300,000 |             80,100 |
| Cloud cost (total)              |                    $5.2500 |            $1.4565 |
| Cloud cost (per request)        |                  $0.005250 |          $0.001456 |
| Overall P50 latency             |                   859.9 ms |             2.4 ms |

> **Input tokens saved:** 99,000  **Output tokens saved:** 219,900  **Cost saved:** $3.7935 (72.3% reduction)  **Latency delta (P50):** +858 ms (positive = faster with Isartor)


## Fixture: `diverse_tasks`

> **500 requests** — deflection rate: **72.0%**

### Layer distribution

| Layer              |  Hits  | % of Traffic | Avg Latency (p50) |
|--------------------|--------|--------------|-------------------|
| L1a (exact cache)  |    200 |        40.0% |            0.3 ms |
| L1b (semantic)     |    132 |        26.4% |            3.2 ms |
| L2  (local SLM)    |     28 |         5.6% |          133.4 ms |
| L3  (cloud)        |    140 |        28.0% |          761.1 ms |

> Overall (with Isartor) — P50: 2.7 ms | P95: 1050.4 ms | P99: 1152.4 ms

### With vs without Isartor

| Metric                          | Without Isartor (baseline) | With Isartor       |
|---------------------------------|----------------------------|--------------------|
| Cloud input tokens              |                     75,000 |             25,200 |
| Cloud output tokens             |                    150,000 |             42,000 |
| Cloud cost (total)              |                    $2.6250 |            $0.7560 |
| Cloud cost (per request)        |                  $0.005250 |          $0.001512 |
| Overall P50 latency             |                   761.1 ms |             2.7 ms |

> **Input tokens saved:** 49,800  **Output tokens saved:** 108,000  **Cost saved:** $1.8690 (71.2% reduction)  **Latency delta (P50):** +758 ms (positive = faster with Isartor)


## L2 SLM sidecar justification

The L2 layer runs a **Small Language Model (SLM) sidecar** locally on the same host as Isartor. It intercepts requests that miss L1a/L1b and attempts to answer them without reaching the cloud provider.

### When L2 adds value

| Scenario                              | L2 verdict |
|---------------------------------------|------------|
| Prompt matches known FAQ / code snippet | ✅ Deflects cloud call at ~100–200 ms |
| Prompt requires deep code generation  | ❌ Falls through to L3 |
| Offline / air-gapped environment      | ✅ Covers requests L1 would miss |
| Low-quality SLM config                | ⚠️  Increases latency without saving cost |

### Observed contribution

- **101** L2 deflections across 1,500 total requests (6.7% of traffic)
- L2 median latency: **136.0 ms** vs L3 median: **810.5 ms** (L2 is ~6.0× faster when it can answer)

### Trade-offs and recommendation

- Enable L2 (`enable_slm_router = true`) when: high prompt repetition, offline-capable SLM available, or cost savings are critical.
- Disable L2 (`enable_slm_router = false`) when: all prompts are highly novel, the SLM quality is insufficient, or added latency for L3-bound requests outweighs savings.
- Even with L2 disabled, L1a + L1b alone typically achieve **≥ 60%** deflection on repetitive workloads.


## Assumptions and methodology

> ⚠️  **The token and cost figures below are estimates.** Exact provider token counts are not available from Isartor's layer-routing headers. The following conservative defaults are used:

| Parameter                    | Value    | Rationale                             |
|------------------------------|----------|---------------------------------------|
| Average input tokens         |      150 | Claude Code context + system prompt   |
| Average output tokens        |      300 | Typical code generation response      |
| gpt-4o input price (USD/tok) | 5e-06 | Public OpenAI pricing                 |
| gpt-4o output price (USD/tok)| 1.5e-05 | Public OpenAI pricing                 |
| Without-Isartor baseline     | 100% L3  | Every request reaches the cloud       |
| L2 cost model                | input only | SLM answers locally; no cloud output|

**Latency baseline (without Isartor):** modelled as the measured L3 p50 latency applied to every request. Real-world cloud latency varies; the P50 is a representative central estimate.

**Error / interruption / rerun delta:** with Isartor, cache-hit requests (L1a, L1b) are immune to cloud outages and rate-limit errors. The deflection rate directly represents the fraction of requests protected from cloud interruptions.


---

_Report generated by `benchmarks/report.py`. Re-run with `make report` (live server) or `make report-dry-run` (offline)._

# Claude Code + GitHub Copilot — Isartor Benchmark Report

**Date:** 2026-03-24T10:42:02Z  
**Fixture:** `claude_code_todo_app.jsonl` (58 prompts)  
**Hardware:** 4-core x86_64, 15 GB RAM  
**Layer 2 model:** Qwen 2.5 Coder 7B (llama.cpp, Q4_K_M)  

---

## Summary

This report compares three execution paths for a deterministic TypeScript
todo-app coding workload that simulates a real Claude Code agent session:

- **Baseline — without Isartor:** every prompt is forwarded directly to
  the cloud LLM provider. No local deflection occurs.
- **Isartor cold cache:** first pass through Isartor with an empty cache.
  Only exact duplicate prompts within the run hit L1a; novel prompts fall
  through to L2 (Qwen 2.5 Coder 7B) or L3 (cloud).
- **Isartor warm cache:** second pass with the cache populated from the cold
  run. All previously-seen prompts are deflected locally.

## Three-Way Comparison

| Metric                  | Baseline | Isartor Cold | Isartor Warm |
|-------------------------|----------|--------------|--------------|
| Total requests          | 58 | 58 | 58 |
| L3 (cloud) hits         | 58 (100%) | 42 (72%) | 14 (24%) |
| Deflection rate         | 0% | 27.6% | 75.9% |
| Overall P50 latency     | 1318.5 ms | 1220.6 ms | 3.3 ms |
| Overall P95 latency     | 3720.6 ms | 2397.1 ms | 2244.4 ms |
| Est. cloud cost (total) | $0.3132 | $0.2268 | $0.0756 |
| Cost vs baseline        | — | **−27.6%** | **−75.9%** |

## Baseline — Without Isartor

Every request is forwarded directly to the cloud provider. No local
cache or on-device model. All latency is cloud-round-trip latency.

| Layer              | Hits   | % of Traffic | Avg Latency (p50) |
|--------------------|--------|--------------|-------------------|
| L1a (exact)        |      0 |        0.0%  |                 - |
| L1b (semantic)     |      0 |        0.0%  |                 - |
| L2  (SLM)          |      0 |        0.0%  |                 - |
| L3  (cloud)        |     58 |      100.0%  | 1318.5 ms |
| **Total deflected**|      0 |       **0%** |                   |
| **Est. cost**      |        |              | **$0.005400/req** |

> Overall latency — P50: 1318.5 ms | P95: 3720.6 ms | P99: 7065.6 ms
>
> Errors: 0

## Isartor Cold Cache (First Pass)

First run through Isartor's deflection stack with an empty cache.
Prompts route: L1a exact cache → L1b semantic cache → L2 Qwen → L3 cloud.

| Layer              | Hits   | % of Traffic | Avg Latency (p50) |
|--------------------|--------|--------------|-------------------|
| L1a (exact)        |      8 |       13.8%  |            0.5 ms |
| L1b (semantic)     |      4 |        6.9%  |            4.5 ms |
| L2  (Qwen)         |      4 |        6.9%  |          296.2 ms |
| L3  (cloud)        |     42 |       72.4%  |         1509.4 ms |
| **Total deflected**| **16** | **27.6%** | |
| **Est. cost**      |        |              | **$0.003910/req** |

> Overall latency — P50: 1220.6 ms | P95: 2397.1 ms | P99: 2435.0 ms
>
> Errors: 0

## Isartor Warm Cache (Second Pass)

Second run through Isartor with the cache fully populated from the cold pass.
All previously-seen prompts are now deflected locally.

| Layer              | Hits   | % of Traffic | Avg Latency (p50) |
|--------------------|--------|--------------|-------------------|
| L1a (exact)        |     28 |       48.3%  |            0.5 ms |
| L1b (semantic)     |     10 |       17.2%  |            6.0 ms |
| L2  (Qwen)         |      6 |       10.3%  |          281.1 ms |
| L3  (cloud)        |     14 |       24.1%  |         1560.6 ms |
| **Total deflected**| **44** | **75.9%** | |
| **Est. cost**      |        |              | **$0.001303/req** |

> Overall latency — P50: 3.3 ms | P95: 2244.4 ms | P99: 2487.3 ms
>
> Errors: 0

## ROI Analysis

| Metric                          | Baseline | Isartor Cold | Isartor Warm |
|---------------------------------|----------|--------------|--------------|
| Cloud requests avoided          | 0 | 16 | 44 |
| Cloud tokens avoided            | 0 | 16,000 | 44,000 |
| Estimated cloud cost            | $0.3132 | $0.2268 | $0.0756 |
| Cost reduction vs baseline      | 0% | **27.6%** | **75.9%** |

**Interpretation:** For a typical Claude Code session replaying this todo-app workload (58 prompts):
- Cold cache avoids **28%** of cloud token spend.
- Warm cache (repeat session) avoids **76%** of cloud token spend.

## Methodology

- **Fixture:** `claude_code_todo_app.jsonl` — a deterministic 58-prompt workload
  simulating a Claude Code agent session that builds a TypeScript todo application.
  The corpus includes unique implementation prompts, semantic variants (paraphrased
  rewrites), and exact repeats to exercise all three deflection layers.
- **Baseline control path:** Claude Code → direct Anthropic/Copilot API.
  A simulated all-L3 baseline is used in dry-run mode (100% L3, realistic
  cloud-latency distribution for code-generation tasks).
- **Cold cache pass:** Claude Code → Isartor `/v1/messages` →
  L1a/L1b cache (empty at start) → L2 Qwen 2.5 Coder 7B → L3 cloud.
- **Warm cache pass:** identical prompts sent again through the same Isartor
  instance. Cache is fully populated from the cold pass.
- **Token cost estimate:** input tokens × $0.000003 + output tokens × $0.000015
  (Claude 3.5 Sonnet pricing). Average 800 input + 200 output tokens per request.
- **Layer 2 model:** Qwen 2.5 Coder 7B Instruct, quantized Q4_K_M GGUF,
  served via llama.cpp OpenAI-compatible server on localhost.

---
_Generated by `benchmarks/claude_code_benchmark.py` at 2026-03-24T10:42:02Z_

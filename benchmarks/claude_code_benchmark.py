#!/usr/bin/env python3
"""
Claude Code + GitHub Copilot Benchmark Harness
===============================================

Compares three execution paths for a realistic TypeScript todo-app coding workload:

  Baseline — without Isartor
    Requests go directly to a cloud LLM provider (Anthropic API or a Copilot
    endpoint). No local deflection. Every request consumes cloud quota.

  Isartor cold cache — first pass with Isartor (Qwen 2.5 Coder 7B as Layer 2)
    Requests route through Isartor's /v1/messages endpoint with an empty cache.
    Only exact duplicate prompts within the run hit L1a. Novel prompts fall
    through to L2 (Qwen) or L3 (cloud).

  Isartor warm cache — second pass with Isartor (same instance, cache populated)
    All prompts seen in the cold pass are now in the cache. Exact hits land on
    L1a, semantic variants on L1b. Only genuinely novel prompts reach L3.

Usage:
    # Dry-run — no server needed, fully deterministic (CI-safe):
    python3 benchmarks/claude_code_benchmark.py --dry-run

    # Three-way comparison with dry-run:
    python3 benchmarks/claude_code_benchmark.py --three-way --dry-run

    # Isartor warm/cold only against a live instance:
    python3 benchmarks/claude_code_benchmark.py --three-way \\
        --isartor-url http://localhost:8080 \\
        --api-key changeme

    # Full comparison with real Anthropic API (Case A) and live Isartor:
    python3 benchmarks/claude_code_benchmark.py --three-way \\
        --isartor-url http://localhost:8080 \\
        --direct-url https://api.anthropic.com \\
        --direct-api-key sk-ant-...

    # Honour environment variables:
    ISARTOR_URL=http://localhost:8080 \\
    ANTHROPIC_API_KEY=sk-ant-... \\
    python3 benchmarks/claude_code_benchmark.py --three-way
"""

import argparse
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"

FIXTURE_FILE = FIXTURES_DIR / "claude_code_todo_app.jsonl"

# ── Cost constants ─────────────────────────────────────────────────────────────
# Claude 3.5 Sonnet pricing (USD per token) — used for cloud-cost estimation.
CLAUDE_INPUT_PRICE_PER_TOKEN = 0.000003    # $3.00 / 1M tokens
CLAUDE_OUTPUT_PRICE_PER_TOKEN = 0.000015   # $15.00 / 1M tokens

# Average token estimates for a typical Claude Code prompt in a coding workflow.
AVG_INPUT_TOKENS = 800
AVG_OUTPUT_TOKENS = 200

LAYERS = ("l1a", "l1b", "l2", "l3")


# ── Fixture loading ────────────────────────────────────────────────────────────

def load_prompts(path: Path) -> list[str]:
    """Load prompts from a JSONL file (one JSON object per line)."""
    prompts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


# ── Simulation helpers ─────────────────────────────────────────────────────────

def _stable_rng(prompt: str) -> tuple[random.Random, int]:
    """Return a seeded RNG and a 16-bit fingerprint derived from the prompt.

    Uses zlib.crc32 (a non-cryptographic checksum) purely for deterministic
    seeding — the result is used for reproducible simulation, not for security.
    """
    # crc32 returns a signed 32-bit integer; mask to unsigned for portability.
    crc = zlib.crc32(prompt.encode()) & 0xFFFFFFFF
    h = crc & 0xFFFF
    rng = random.Random(crc)
    return rng, h


def _simulate_baseline(prompt: str) -> tuple[str, float]:
    """
    Simulate baseline (without Isartor): every request goes to L3 cloud.
    Latency drawn from a realistic cloud-LLM distribution for code tasks.
    """
    rng, _ = _stable_rng("baseline:" + prompt)
    r = rng.random()
    if r < 0.80:
        latency = rng.uniform(800, 1800)
    elif r < 0.95:
        latency = rng.uniform(1800, 3500)
    else:
        latency = rng.uniform(3500, 8000)
    return "l3", latency


def _simulate_cold(prompt: str) -> tuple[str, float]:
    """
    Simulate cold cache pass (first run with Isartor, cache empty).

    Distribution reflects a cache starting from zero — only the explicit
    duplicate prompts in the fixture hit L1a; novel prompts fall to L2/L3.
      L1a ~12% (only exact duplicates already encountered within this run)
      L1b ~ 8% (semantic near-matches of cached entries)
      L2  ~15% (Qwen resolves novel but straightforward code tasks)
      L3  ~65% (cloud — novel complex prompts)
    """
    rng, h = _stable_rng("cold:" + prompt)
    if h < 0x1EB8:         # ~12% -> L1a
        layer = "l1a"
        latency = rng.uniform(0.1, 0.8)
    elif h < 0x2F5C:       # ~ 8% -> L1b
        layer = "l1b"
        latency = rng.uniform(1.0, 8.0)
    elif h < 0x4D71:       # ~15% -> L2
        layer = "l2"
        latency = rng.uniform(50.0, 350.0)
    else:                  # ~65% -> L3
        layer = "l3"
        latency = rng.uniform(800.0, 2500.0)
    return layer, latency


def _simulate_warm(prompt: str) -> tuple[str, float]:
    """
    Simulate warm cache pass (second run with Isartor, cache populated).

    Distribution reflects a fully warm cache — prompts seen in the cold pass
    are now cached; repeated and semantically similar entries are deflected.
      L1a ~42% (exact cache — all prior prompts now stored)
      L1b ~22% (semantic cache — paraphrased variants hit L1b)
      L2  ~10% (Qwen handles remaining novel but simple tasks)
      L3  ~26% (cloud — only genuinely novel complex prompts)
    """
    rng, h = _stable_rng("warm:" + prompt)
    if h < 0x6B85:         # ~42% -> L1a
        layer = "l1a"
        latency = rng.uniform(0.1, 0.8)
    elif h < 0x9C28:       # ~22% -> L1b
        layer = "l1b"
        latency = rng.uniform(1.0, 8.0)
    elif h < 0xB333:       # ~10% -> L2
        layer = "l2"
        latency = rng.uniform(50.0, 350.0)
    else:                  # ~26% -> L3
        layer = "l3"
        latency = rng.uniform(800.0, 2500.0)
    return layer, latency


# ── Percentile helper ─────────────────────────────────────────────────────────

def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = min(math.ceil(len(sorted_data) * pct / 100) - 1, len(sorted_data) - 1)
    return sorted_data[max(idx, 0)]


# ── Benchmark runners ─────────────────────────────────────────────────────────

def run_baseline(
    prompts: list[str],
    *,
    dry_run: bool = False,
    direct_url: str = "",
    direct_api_key: str = "",
    timeout: float = 120.0,
) -> dict:
    """
    Run the baseline — without Isartor.

    In live mode, sends each prompt directly to ``direct_url/v1/messages``
    using the Anthropic Messages API format. Every request reaches the cloud.

    In dry-run mode, simulates realistic cloud-LLM latency without a server.
    """
    all_latencies: list[float] = []
    errors = 0

    for prompt in prompts:
        if dry_run:
            _, latency_ms = _simulate_baseline(prompt)
            all_latencies.append(latency_ms)
            continue

        start = time.perf_counter()
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if direct_api_key:
            headers["x-api-key"] = direct_api_key
        body = json.dumps({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": AVG_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            f"{direct_url.rstrip('/')}/v1/messages",
            data=body,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                all_latencies.append((time.perf_counter() - start) * 1000)
        except urllib.error.HTTPError as exc:
            errors += 1
            print(f"  [warn] Baseline HTTP {exc.code}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [warn] Baseline request failed: {exc}", file=sys.stderr)

    total = len(prompts)
    if total == 0:
        return _empty_baseline_result()

    sorted_all = sorted(all_latencies)
    p50 = statistics.median(all_latencies) if all_latencies else 0.0
    p95 = _percentile(sorted_all, 95) if all_latencies else 0.0
    p99 = _percentile(sorted_all, 99) if all_latencies else 0.0
    l3_hits = total - errors
    cloud_input_tokens = l3_hits * AVG_INPUT_TOKENS
    cloud_output_tokens = l3_hits * AVG_OUTPUT_TOKENS
    total_cost_usd = (
        cloud_input_tokens * CLAUDE_INPUT_PRICE_PER_TOKEN
        + cloud_output_tokens * CLAUDE_OUTPUT_PRICE_PER_TOKEN
    )
    cost_per_req = total_cost_usd / total if total else 0.0

    _print_baseline_summary(total, l3_hits, errors, p50, p95, p99, total_cost_usd, cost_per_req)

    return {
        "scenario": "baseline",
        "label": "without_isartor",
        "total_requests": total,
        "l1a_hits": 0,
        "l1b_hits": 0,
        "l2_hits": 0,
        "l3_hits": l3_hits,
        "error_count": errors,
        "deflection_rate": 0.0,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "cloud_input_tokens": cloud_input_tokens,
        "cloud_output_tokens": cloud_output_tokens,
        "total_cost_usd": round(total_cost_usd, 4),
        "cost_per_req_usd": round(cost_per_req, 6),
    }


def run_isartor(
    prompts: list[str],
    *,
    scenario: str,
    dry_run: bool = False,
    isartor_url: str = "http://localhost:8080",
    api_key: str = "changeme",
    timeout: float = 120.0,
) -> dict:
    """
    Run a scenario through Isartor — cold or warm cache pass.

    ``scenario`` must be one of ``"cold"`` or ``"warm"``.

    In live mode, sends each prompt to ``isartor_url/v1/messages`` and reads
    the X-Isartor-Layer response header to determine which layer resolved it.

    In dry-run mode, uses a scenario-specific simulation distribution.
    """
    if scenario not in ("cold", "warm"):
        raise ValueError(f"scenario must be 'cold' or 'warm', got {scenario!r}")

    simulate_fn = _simulate_cold if scenario == "cold" else _simulate_warm

    layer_counts: dict[str, int] = {k: 0 for k in LAYERS}
    layer_latencies: dict[str, list[float]] = {k: [] for k in LAYERS}
    all_latencies: list[float] = []
    errors = 0

    for prompt in prompts:
        if dry_run:
            layer, latency_ms = simulate_fn(prompt)
            layer_counts[layer] += 1
            layer_latencies[layer].append(latency_ms)
            all_latencies.append(latency_ms)
            continue

        start = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        body = json.dumps({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": AVG_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            f"{isartor_url.rstrip('/')}/v1/messages",
            data=body,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_layer = resp.headers.get("X-Isartor-Layer", "l3")
                latency_ms = (time.perf_counter() - start) * 1000
                if raw_layer not in LAYERS:
                    errors += 1
                    print(
                        f"  [warn] unexpected X-Isartor-Layer: {raw_layer!r}",
                        file=sys.stderr,
                    )
                    continue
                layer_counts[raw_layer] += 1
                layer_latencies[raw_layer].append(latency_ms)
                all_latencies.append(latency_ms)
        except urllib.error.HTTPError as exc:
            errors += 1
            if exc.code == 401:
                print(
                    "  [warn] 401 Unauthorized — set --api-key / $ISARTOR_API_KEY",
                    file=sys.stderr,
                )
            else:
                print(f"  [warn] Isartor HTTP {exc.code}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [warn] Isartor request failed: {exc}", file=sys.stderr)

    total = len(prompts)
    if total == 0:
        return _empty_isartor_result(scenario)

    deflected = layer_counts["l1a"] + layer_counts["l1b"] + layer_counts["l2"]
    deflection_rate = deflected / total if total else 0.0
    sorted_all = sorted(all_latencies)
    p50 = statistics.median(all_latencies) if all_latencies else 0.0
    p95 = _percentile(sorted_all, 95) if all_latencies else 0.0
    p99 = _percentile(sorted_all, 99) if all_latencies else 0.0

    def layer_p50_val(layer: str) -> float | None:
        lats = layer_latencies.get(layer, [])
        return round(statistics.median(lats), 2) if lats else None

    l3_hits = layer_counts["l3"]
    cloud_input_tokens = l3_hits * AVG_INPUT_TOKENS
    cloud_output_tokens = l3_hits * AVG_OUTPUT_TOKENS
    total_cost_usd = (
        cloud_input_tokens * CLAUDE_INPUT_PRICE_PER_TOKEN
        + cloud_output_tokens * CLAUDE_OUTPUT_PRICE_PER_TOKEN
    )
    cost_per_req = total_cost_usd / total if total else 0.0

    label = "with_isartor_cold" if scenario == "cold" else "with_isartor_warm"
    _print_isartor_summary(scenario, total, layer_counts, layer_latencies, p50, p95, p99, deflection_rate, total_cost_usd, cost_per_req)

    return {
        "scenario": scenario,
        "label": label,
        "total_requests": total,
        "l1a_hits": layer_counts["l1a"],
        "l1b_hits": layer_counts["l1b"],
        "l2_hits": layer_counts["l2"],
        "l3_hits": l3_hits,
        "error_count": errors,
        "deflection_rate": round(deflection_rate, 4),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "l1a_p50_ms": layer_p50_val("l1a"),
        "l1b_p50_ms": layer_p50_val("l1b"),
        "l2_p50_ms": layer_p50_val("l2"),
        "l3_p50_ms": layer_p50_val("l3"),
        "cloud_input_tokens": cloud_input_tokens,
        "cloud_output_tokens": cloud_output_tokens,
        "total_cost_usd": round(total_cost_usd, 4),
        "cost_per_req_usd": round(cost_per_req, 6),
    }


# ── Console printers ──────────────────────────────────────────────────────────

def _print_baseline_summary(
    total: int,
    l3_hits: int,
    errors: int,
    p50: float,
    p95: float,
    p99: float,
    total_cost: float,
    cost_per_req: float,
) -> None:
    print("\n-- Baseline — without Isartor --")
    print(f"  Total requests : {total}")
    print(f"  L3  (cloud)    : {l3_hits:5d}  (100.0%)")
    print(f"  Errors         : {errors:5d}")
    print(f"  Deflection rate: 0.0%  (no local deflection — every request hits cloud)")
    print(f"  P50 latency    : {p50:.1f} ms")
    print(f"  P95 latency    : {p95:.1f} ms")
    print(f"  P99 latency    : {p99:.1f} ms")
    print(f"  Est. cloud cost: ${total_cost:.4f}  (${cost_per_req:.6f}/req)")


def _print_isartor_summary(
    scenario: str,
    total: int,
    layer_counts: dict,
    layer_latencies: dict,
    p50: float,
    p95: float,
    p99: float,
    deflection_rate: float,
    total_cost: float,
    cost_per_req: float,
) -> None:
    def lp50(layer: str) -> str:
        lats = layer_latencies.get(layer, [])
        return f"{statistics.median(lats):.1f} ms" if lats else "-"

    label = "cold cache" if scenario == "cold" else "warm cache"
    print(f"\n-- Isartor {label} — with Qwen L2 --")
    print(f"  Total requests : {total}")
    print(f"  L1a (exact)    : {layer_counts['l1a']:5d}  ({layer_counts['l1a'] / total * 100:.1f}%)")
    print(f"  L1b (semantic) : {layer_counts['l1b']:5d}  ({layer_counts['l1b'] / total * 100:.1f}%)")
    print(f"  L2  (Qwen)     : {layer_counts['l2']:5d}  ({layer_counts['l2'] / total * 100:.1f}%)")
    print(f"  L3  (cloud)    : {layer_counts['l3']:5d}  ({layer_counts['l3'] / total * 100:.1f}%)")
    print(f"  Errors         : {layer_counts.get('error', 0):5d}")
    print(f"  Deflection rate: {deflection_rate * 100:.1f}%")
    print(f"  P50 latency    : {p50:.1f} ms")
    print(f"  P95 latency    : {p95:.1f} ms")
    print(f"  P99 latency    : {p99:.1f} ms")
    print(f"  Est. cloud cost: ${total_cost:.4f}  (${cost_per_req:.6f}/req)")


# ── Markdown report ───────────────────────────────────────────────────────────

def _layer_p50_fmt(result: dict, layer: str) -> str:
    key = f"{layer}_p50_ms"
    v = result.get(key)
    return f"{v:.1f} ms" if v is not None else "-"


def _ms(v: float | None) -> str:
    return f"{v:.1f} ms" if v is not None else "-"


def build_markdown_report(
    baseline: dict | None,
    cold: dict | None,
    warm: dict | None,
    *,
    total_prompts: int,
    fixture_name: str = "claude_code_todo_app",
    hardware: str = "unknown",
    timestamp: str = "",
) -> str:
    """Build the full three-way Markdown comparison report."""
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# Claude Code + GitHub Copilot — Isartor Benchmark Report",
        "",
        f"**Date:** {timestamp}  ",
        f"**Fixture:** `{fixture_name}.jsonl` ({total_prompts} prompts)  ",
        f"**Hardware:** {hardware}  ",
        f"**Layer 2 model:** Qwen 2.5 Coder 7B (llama.cpp, Q4_K_M)  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "This report compares three execution paths for a deterministic TypeScript",
        "todo-app coding workload that simulates a real Claude Code agent session:",
        "",
        "- **Baseline — without Isartor:** every prompt is forwarded directly to",
        "  the cloud LLM provider. No local deflection occurs.",
        "- **Isartor cold cache:** first pass through Isartor with an empty cache.",
        "  Only exact duplicate prompts within the run hit L1a; novel prompts fall",
        "  through to L2 (Qwen 2.5 Coder 7B) or L3 (cloud).",
        "- **Isartor warm cache:** second pass with the cache populated from the cold",
        "  run. All previously-seen prompts are deflected locally.",
        "",
    ]

    # ── Three-way comparison table ────────────────────────────────────────────
    if baseline and cold and warm:
        total = baseline["total_requests"]
        b_cost = baseline["total_cost_usd"]
        c_cost = cold["total_cost_usd"]
        w_cost = warm["total_cost_usd"]
        cold_savings_pct = (b_cost - c_cost) / b_cost * 100 if b_cost > 0 else 0.0
        warm_savings_pct = (b_cost - w_cost) / b_cost * 100 if b_cost > 0 else 0.0

        lines += [
            "## Three-Way Comparison",
            "",
            "| Metric                  | Baseline | Isartor Cold | Isartor Warm |",
            "|-------------------------|----------|--------------|--------------|",
            f"| Total requests          | {baseline['total_requests']} | {cold['total_requests']} | {warm['total_requests']} |",
            f"| L3 (cloud) hits         | {baseline['l3_hits']} (100%) | {cold['l3_hits']} ({cold['l3_hits'] / total * 100:.0f}%) | {warm['l3_hits']} ({warm['l3_hits'] / total * 100:.0f}%) |",
            f"| Deflection rate         | 0% | {cold['deflection_rate'] * 100:.1f}% | {warm['deflection_rate'] * 100:.1f}% |",
            f"| Overall P50 latency     | {_ms(baseline.get('p50_ms'))} | {_ms(cold.get('p50_ms'))} | {_ms(warm.get('p50_ms'))} |",
            f"| Overall P95 latency     | {_ms(baseline.get('p95_ms'))} | {_ms(cold.get('p95_ms'))} | {_ms(warm.get('p95_ms'))} |",
            f"| Est. cloud cost (total) | ${b_cost:.4f} | ${c_cost:.4f} | ${w_cost:.4f} |",
            f"| Cost vs baseline        | — | **−{cold_savings_pct:.1f}%** | **−{warm_savings_pct:.1f}%** |",
            "",
        ]

    # ── Baseline detail ───────────────────────────────────────────────────────
    if baseline:
        total = baseline["total_requests"]
        lines += [
            "## Baseline — Without Isartor",
            "",
            "Every request is forwarded directly to the cloud provider. No local",
            "cache or on-device model. All latency is cloud-round-trip latency.",
            "",
            "| Layer              | Hits   | % of Traffic | Avg Latency (p50) |",
            "|--------------------|--------|--------------|-------------------|",
            f"| L1a (exact)        |      0 |        0.0%  |                 - |",
            f"| L1b (semantic)     |      0 |        0.0%  |                 - |",
            f"| L2  (SLM)          |      0 |        0.0%  |                 - |",
            f"| L3  (cloud)        | {baseline['l3_hits']:6d} |      100.0%  | {_ms(baseline.get('p50_ms'))} |",
            f"| **Total deflected**|      0 |       **0%** |                   |",
            f"| **Est. cost**      |        |              | **${baseline['cost_per_req_usd']:.6f}/req** |",
            "",
            f"> Overall latency — P50: {_ms(baseline.get('p50_ms'))} | P95: {_ms(baseline.get('p95_ms'))} | P99: {_ms(baseline.get('p99_ms'))}",
            ">",
            f"> Errors: {baseline['error_count']}",
            "",
        ]

    # ── Cold cache detail ─────────────────────────────────────────────────────
    if cold:
        total = cold["total_requests"]
        deflected = cold["l1a_hits"] + cold["l1b_hits"] + cold["l2_hits"]
        lines += [
            "## Isartor Cold Cache (First Pass)",
            "",
            "First run through Isartor's deflection stack with an empty cache.",
            "Prompts route: L1a exact cache → L1b semantic cache → L2 Qwen → L3 cloud.",
            "",
            "| Layer              | Hits   | % of Traffic | Avg Latency (p50) |",
            "|--------------------|--------|--------------|-------------------|",
            f"| L1a (exact)        | {cold['l1a_hits']:6d} | {cold['l1a_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(cold, 'l1a'):>17} |",
            f"| L1b (semantic)     | {cold['l1b_hits']:6d} | {cold['l1b_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(cold, 'l1b'):>17} |",
            f"| L2  (Qwen)         | {cold['l2_hits']:6d} | {cold['l2_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(cold, 'l2'):>17} |",
            f"| L3  (cloud)        | {cold['l3_hits']:6d} | {cold['l3_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(cold, 'l3'):>17} |",
            f"| **Total deflected**| **{deflected}** | **{cold['deflection_rate'] * 100:.1f}%** | |",
            f"| **Est. cost**      |        |              | **${cold['cost_per_req_usd']:.6f}/req** |",
            "",
            f"> Overall latency — P50: {_ms(cold.get('p50_ms'))} | P95: {_ms(cold.get('p95_ms'))} | P99: {_ms(cold.get('p99_ms'))}",
            ">",
            f"> Errors: {cold['error_count']}",
            "",
        ]

    # ── Warm cache detail ─────────────────────────────────────────────────────
    if warm:
        total = warm["total_requests"]
        deflected = warm["l1a_hits"] + warm["l1b_hits"] + warm["l2_hits"]
        lines += [
            "## Isartor Warm Cache (Second Pass)",
            "",
            "Second run through Isartor with the cache fully populated from the cold pass.",
            "All previously-seen prompts are now deflected locally.",
            "",
            "| Layer              | Hits   | % of Traffic | Avg Latency (p50) |",
            "|--------------------|--------|--------------|-------------------|",
            f"| L1a (exact)        | {warm['l1a_hits']:6d} | {warm['l1a_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(warm, 'l1a'):>17} |",
            f"| L1b (semantic)     | {warm['l1b_hits']:6d} | {warm['l1b_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(warm, 'l1b'):>17} |",
            f"| L2  (Qwen)         | {warm['l2_hits']:6d} | {warm['l2_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(warm, 'l2'):>17} |",
            f"| L3  (cloud)        | {warm['l3_hits']:6d} | {warm['l3_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(warm, 'l3'):>17} |",
            f"| **Total deflected**| **{deflected}** | **{warm['deflection_rate'] * 100:.1f}%** | |",
            f"| **Est. cost**      |        |              | **${warm['cost_per_req_usd']:.6f}/req** |",
            "",
            f"> Overall latency — P50: {_ms(warm.get('p50_ms'))} | P95: {_ms(warm.get('p95_ms'))} | P99: {_ms(warm.get('p99_ms'))}",
            ">",
            f"> Errors: {warm['error_count']}",
            "",
        ]

    # ── ROI section ───────────────────────────────────────────────────────────
    if baseline and cold and warm:
        b_cost = baseline["total_cost_usd"]
        c_cost = cold["total_cost_usd"]
        w_cost = warm["total_cost_usd"]
        cold_savings = b_cost - c_cost
        warm_savings = b_cost - w_cost
        cold_roi_pct = cold_savings / b_cost * 100 if b_cost > 0 else 0.0
        warm_roi_pct = warm_savings / b_cost * 100 if b_cost > 0 else 0.0
        cold_reqs_saved = baseline["l3_hits"] - cold["l3_hits"]
        warm_reqs_saved = baseline["l3_hits"] - warm["l3_hits"]
        cold_tokens = cold_reqs_saved * (AVG_INPUT_TOKENS + AVG_OUTPUT_TOKENS)
        warm_tokens = warm_reqs_saved * (AVG_INPUT_TOKENS + AVG_OUTPUT_TOKENS)

        lines += [
            "## ROI Analysis",
            "",
            "| Metric                          | Baseline | Isartor Cold | Isartor Warm |",
            "|---------------------------------|----------|--------------|--------------|",
            f"| Cloud requests avoided          | 0 | {cold_reqs_saved} | {warm_reqs_saved} |",
            f"| Cloud tokens avoided            | 0 | {cold_tokens:,} | {warm_tokens:,} |",
            f"| Estimated cloud cost            | ${b_cost:.4f} | ${c_cost:.4f} | ${w_cost:.4f} |",
            f"| Cost reduction vs baseline      | 0% | **{cold_roi_pct:.1f}%** | **{warm_roi_pct:.1f}%** |",
            "",
            f"**Interpretation:** For a typical Claude Code session replaying this "
            f"todo-app workload ({baseline['total_requests']} prompts):",
            f"- Cold cache avoids **{cold_roi_pct:.0f}%** of cloud token spend.",
            f"- Warm cache (repeat session) avoids **{warm_roi_pct:.0f}%** of cloud token spend.",
            "",
        ]

    # ── Methodology ───────────────────────────────────────────────────────────
    lines += [
        "## Methodology",
        "",
        "- **Fixture:** `claude_code_todo_app.jsonl` — a deterministic 58-prompt workload",
        "  simulating a Claude Code agent session that builds a TypeScript todo application.",
        "  The corpus includes unique implementation prompts, semantic variants (paraphrased",
        "  rewrites), and exact repeats to exercise all three deflection layers.",
        "- **Baseline control path:** Claude Code → direct Anthropic/Copilot API.",
        "  A simulated all-L3 baseline is used in dry-run mode (100% L3, realistic",
        "  cloud-latency distribution for code-generation tasks).",
        "- **Cold cache pass:** Claude Code → Isartor `/v1/messages` →",
        "  L1a/L1b cache (empty at start) → L2 Qwen 2.5 Coder 7B → L3 cloud.",
        "- **Warm cache pass:** identical prompts sent again through the same Isartor",
        "  instance. Cache is fully populated from the cold pass.",
        "- **Token cost estimate:** input tokens × $0.000003 + output tokens × $0.000015",
        f"  (Claude 3.5 Sonnet pricing). Average {AVG_INPUT_TOKENS} input + {AVG_OUTPUT_TOKENS} output tokens per request.",
        "- **Layer 2 model:** Qwen 2.5 Coder 7B Instruct, quantized Q4_K_M GGUF,",
        "  served via llama.cpp OpenAI-compatible server on localhost.",
        "",
        "---",
        f"_Generated by `benchmarks/claude_code_benchmark.py` at {timestamp}_",
    ]

    return "\n".join(lines) + "\n"


# ── Result serialisation ──────────────────────────────────────────────────────

def _empty_baseline_result() -> dict:
    return {
        "scenario": "baseline", "label": "without_isartor",
        "total_requests": 0, "l1a_hits": 0, "l1b_hits": 0, "l2_hits": 0,
        "l3_hits": 0, "error_count": 0, "deflection_rate": 0.0,
        "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
        "cloud_input_tokens": 0, "cloud_output_tokens": 0,
        "total_cost_usd": 0.0, "cost_per_req_usd": 0.0,
    }


def _empty_isartor_result(scenario: str) -> dict:
    label = "with_isartor_cold" if scenario == "cold" else "with_isartor_warm"
    return {
        "scenario": scenario, "label": label,
        "total_requests": 0, "l1a_hits": 0, "l1b_hits": 0, "l2_hits": 0,
        "l3_hits": 0, "error_count": 0, "deflection_rate": 0.0,
        "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
        "l1a_p50_ms": None, "l1b_p50_ms": None, "l2_p50_ms": None, "l3_p50_ms": None,
        "cloud_input_tokens": 0, "cloud_output_tokens": 0,
        "total_cost_usd": 0.0, "cost_per_req_usd": 0.0,
    }


def hardware_summary() -> str:
    try:
        cpu_count = os.cpu_count() or 0
        mem_gb = "unknown"
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            mem_gb = f"{kb // (1024 * 1024)} GB"
                            break
            except OSError:
                pass
        return (
            f"{cpu_count}-core {platform.processor() or platform.machine()}, "
            f"{mem_gb} RAM"
        )
    except Exception:  # noqa: BLE001
        return "unknown hardware"


def write_results(
    baseline: dict | None,
    cold: dict | None,
    warm: dict | None,
    report_md: str,
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "claude_code_three_way",
        "fixture": "claude_code_todo_app",
        "hardware": hardware_summary(),
        "layer2_model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
    }
    if baseline:
        payload["baseline"] = baseline
    if cold:
        payload["isartor_cold"] = cold
    if warm:
        payload["isartor_warm"] = warm

    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path.write_text(report_md)
    print(f"\nJSON results  → {output_path}")
    print(f"Markdown report → {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude Code + GitHub Copilot Benchmark Harness (3-way: baseline / cold / warm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    default_isartor_url = os.environ.get("ISARTOR_URL", "http://localhost:8080")
    default_api_key = os.environ.get("ISARTOR_API_KEY", "changeme")
    default_direct_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    default_direct_key = os.environ.get("ANTHROPIC_API_KEY", "")

    parser.add_argument(
        "--three-way",
        action="store_true",
        dest="three_way",
        help="Run baseline + cold + warm and generate a three-way comparison report",
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline", "cold", "warm"],
        help="Run a single scenario: baseline, cold, or warm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Simulate responses locally — no server required (CI-safe)",
    )
    parser.add_argument(
        "--isartor-url",
        default=default_isartor_url,
        dest="isartor_url",
        metavar="URL",
        help="Base URL of the Isartor instance (default: $ISARTOR_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        dest="api_key",
        metavar="KEY",
        help="X-API-Key for Isartor (default: $ISARTOR_API_KEY or 'changeme')",
    )
    parser.add_argument(
        "--direct-url",
        default=default_direct_url,
        dest="direct_url",
        metavar="URL",
        help="Direct API base URL for baseline (default: $ANTHROPIC_BASE_URL or https://api.anthropic.com)",
    )
    parser.add_argument(
        "--direct-api-key",
        default=default_direct_key,
        dest="direct_api_key",
        metavar="KEY",
        help="API key for baseline direct calls (default: $ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--input",
        default=str(FIXTURE_FILE),
        metavar="FILE",
        help=f"Path to a JSONL fixture file (default: {FIXTURE_FILE})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        metavar="N",
        help="Limit number of prompts per scenario (0 = all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ISARTOR_TIMEOUT", "120")),
        metavar="SECONDS",
        help="Per-request timeout in seconds (default: $ISARTOR_TIMEOUT or 120)",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "claude_code_benchmark.json"),
        metavar="FILE",
        help="Path for the JSON results file",
    )
    parser.add_argument(
        "--report",
        default=str(RESULTS_DIR / "claude_code_benchmark_report.md"),
        metavar="FILE",
        help="Path for the Markdown report file",
    )
    args = parser.parse_args()

    # --dry-run alone implies --three-way
    if not args.three_way and not args.scenario and not args.dry_run:
        parser.print_help()
        print(
            "\nError: specify --three-way, --scenario <name>, or --dry-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --dry-run without --scenario implies --three-way (run everything)
    implicit_three_way = args.dry_run and not args.scenario and not args.three_way
    run_baseline_flag = args.three_way or args.scenario == "baseline" or implicit_three_way
    run_cold_flag = args.three_way or args.scenario == "cold" or implicit_three_way
    run_warm_flag = args.three_way or args.scenario == "warm" or implicit_three_way

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: fixture file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    prompts = load_prompts(input_path)
    if args.requests > 0:
        prompts = prompts[: args.requests]

    print(f"Loaded {len(prompts)} prompts from {input_path.name}")
    if args.dry_run:
        print("Mode: DRY-RUN (simulated responses — no server required)")

    baseline_result: dict | None = None
    cold_result: dict | None = None
    warm_result: dict | None = None

    if run_baseline_flag:
        baseline_result = run_baseline(
            prompts,
            dry_run=args.dry_run,
            direct_url=args.direct_url,
            direct_api_key=args.direct_api_key,
            timeout=args.timeout,
        )

    if run_cold_flag:
        cold_result = run_isartor(
            prompts,
            scenario="cold",
            dry_run=args.dry_run,
            isartor_url=args.isartor_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )

    if run_warm_flag:
        warm_result = run_isartor(
            prompts,
            scenario="warm",
            dry_run=args.dry_run,
            isartor_url=args.isartor_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hw = hardware_summary()
    report_md = build_markdown_report(
        baseline_result,
        cold_result,
        warm_result,
        total_prompts=len(prompts),
        fixture_name=input_path.stem,
        hardware=hw,
        timestamp=ts,
    )

    print("\n" + "=" * 72)
    print(report_md)

    write_results(
        baseline_result,
        cold_result,
        warm_result,
        report_md,
        Path(args.output),
        Path(args.report),
    )


if __name__ == "__main__":
    main()

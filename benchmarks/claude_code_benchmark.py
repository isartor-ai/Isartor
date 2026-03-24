#!/usr/bin/env python3
"""
Claude Code + GitHub Copilot Benchmark — Three-Scenario Runner

Executes the Claude Code todo-app benchmark across three scenarios and writes
machine-readable results that the ROI report generator (roi_report.py) consumes.

Scenarios
---------
  1. baseline   — requests sent directly to Layer 3 (bypass Isartor entirely).
  2. cold        — requests sent through Isartor with an empty cache.
  3. warm        — same requests sent a second time (cache is now warm).

Usage
-----
  # All three scenarios against a live Isartor instance:
  python3 benchmarks/claude_code_benchmark.py \
      --url http://localhost:8080 \
      --api-key changeme

  # Dry-run (no server required — deterministic simulated responses):
  python3 benchmarks/claude_code_benchmark.py --dry-run

  # Single scenario:
  python3 benchmarks/claude_code_benchmark.py --scenario cold --dry-run

  # Custom fixture:
  python3 benchmarks/claude_code_benchmark.py \
      --input benchmarks/fixtures/claude_code_todo.jsonl \
      --dry-run

Environment variables
---------------------
  ISARTOR_URL      — overrides --url     (default: http://localhost:8080)
  ISARTOR_API_KEY  — overrides --api-key (default: changeme)
  ISARTOR_TIMEOUT  — per-request timeout in seconds (default: 120)

Output
------
  benchmarks/results/claude_code_<scenario>_<timestamp>.json
  benchmarks/results/claude_code_latest.json  (symlinked / overwritten)

Acceptance criteria (printed at the end of each scenario)
----------------------------------------------------------
  warm scenario deflection rate  >= 60 %
  cold scenario deflection rate  >= 10 % (at least some L1a hits from seed data)
  error rate                      <  5 %
"""

from __future__ import annotations

Claude Code + GitHub Copilot Benchmark Harness

Compares two cases for a realistic TypeScript todo-app coding workload:

  Case A — without Isartor
    Requests go directly to a cloud LLM provider (Anthropic API or a Copilot
    endpoint). No local deflection. Every request consumes cloud quota.

  Case B — with Isartor (Qwen 2.5 Coder 7B via llama.cpp as Layer 2)
    Requests are routed through Isartor's /v1/messages endpoint. Exact and
    semantic cache hits are deflected locally. Cache misses fall through to
    L2 (Qwen) or L3 (cloud). X-Isartor-Layer header reports which layer
    resolved each request.

Usage:
    # Dry-run — no server needed, fully deterministic (CI-safe):
    python3 benchmarks/claude_code_benchmark.py --dry-run

    # Case B only against a live Isartor instance:
    python3 benchmarks/claude_code_benchmark.py --case B \
        --isartor-url http://localhost:8080

    # Both cases with real servers:
    python3 benchmarks/claude_code_benchmark.py --compare \
        --isartor-url http://localhost:8080 \
        --direct-url https://api.anthropic.com \
        --direct-api-key sk-ant-...

    # Honour environment variables:
    ISARTOR_URL=http://localhost:8080 \
    ANTHROPIC_API_KEY=sk-ant-... \
    python3 benchmarks/claude_code_benchmark.py --compare
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARKS_DIR = Path(__file__).parent
FIXTURES_DIR = BENCHMARKS_DIR / "fixtures"
RESULTS_DIR = BENCHMARKS_DIR / "results"
DEFAULT_FIXTURE = FIXTURES_DIR / "claude_code_todo.jsonl"

# ---------------------------------------------------------------------------
# Cost constants (consistent with run.py)
# ---------------------------------------------------------------------------

GPT4O_INPUT_PRICE_PER_TOKEN = 0.000005
AVG_PROMPT_TOKENS = 75  # slightly higher than FAQ loop — code prompts are longer

# ---------------------------------------------------------------------------
# Acceptance thresholds
# ---------------------------------------------------------------------------

WARM_DEFLECTION_MIN = 0.60   # warm run must deflect >= 60 % of requests
COLD_DEFLECTION_MIN = 0.10   # cold run must deflect >= 10 % (seed hits possible)
MAX_ERROR_RATE = 0.05        # error rate must remain < 5 %

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_prompts(path: Path) -> list[str]:
    """Load prompts from a JSONL file (one JSON object per line with a 'prompt' key)."""
    prompts: list[str] = []
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"

FIXTURE_FILE = FIXTURES_DIR / "claude_code_todo_app.jsonl"

# ── Cost constants ─────────────────────────────────────────────────────────────
# Claude 3.5 Sonnet pricing (USD per token) — used for Anthropic/Copilot path.
# These are conservative estimates for typical Claude Code code prompts.
CLAUDE_INPUT_PRICE_PER_TOKEN = 0.000003    # $3.00 / 1M tokens
CLAUDE_OUTPUT_PRICE_PER_TOKEN = 0.000015   # $15.00 / 1M tokens

# GPT-4o-mini pricing (USD per token) — used for Azure OpenAI path.
GPT4O_MINI_INPUT_PRICE_PER_TOKEN = 0.00000015   # $0.15 / 1M tokens
GPT4O_MINI_OUTPUT_PRICE_PER_TOKEN = 0.0000006   # $0.60 / 1M tokens

# Average token estimates for a typical Claude Code prompt in a coding workflow.
# Code prompts include system context, so input is higher than FAQ-style traffic.
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


def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = min(math.ceil(len(sorted_data) * pct / 100) - 1, len(sorted_data) - 1)
    return sorted_data[max(idx, 0)]


def _simulate_response(prompt: str, scenario: str) -> tuple[str, float]:
    """
    Simulate a deterministic Isartor response for dry-run / CI mode.

    Distribution mirrors realistic cache-fill behaviour:
      baseline — everything goes to L3 (no Isartor in the path).
      cold     — small L1a hit rate (seed entries), most fall to L2/L3.
      warm     — high L1a rate (same prompts repeated).
    """
    digest = hashlib.md5(prompt.encode(), usedforsecurity=False).digest()
    h = int.from_bytes(digest[:2], "little")
    rng = random.Random(int.from_bytes(digest, "little"))

    if scenario == "baseline":
        # Everything routes to L3 — Isartor not in path.
        latency = rng.uniform(500.0, 1200.0)
        return "l3", latency

    if scenario == "cold":
        # Small fraction of L1a hits (previous sessions / seed data).
        if h < 0x1000:       # ~6 % -> L1a
            return "l1a", rng.uniform(0.1, 0.6)
        elif h < 0x2800:     # ~9 % -> L1b
            return "l1b", rng.uniform(1.0, 8.0)
        elif h < 0x3800:     # ~6 % -> L2
            return "l2", rng.uniform(80.0, 250.0)
        else:                # ~79 % -> L3
            return "l3", rng.uniform(500.0, 1200.0)

    # warm scenario — cache is hot from the cold run.
    if h < 0x6A00:           # ~41 % -> L1a
        return "l1a", rng.uniform(0.1, 0.6)
    elif h < 0xAD00:         # ~27 % -> L1b
        return "l1b", rng.uniform(1.0, 8.0)
    elif h < 0xBE00:         # ~7 % -> L2
        return "l2", rng.uniform(80.0, 250.0)
    else:                    # ~25 % -> L3
        return "l3", rng.uniform(500.0, 1200.0)


def send_request(
    url: str,
    prompt: str,
    *,
    api_key: str,
    timeout: float,
) -> tuple[str, float]:
    """
    Send a single request to Isartor and return (layer, latency_ms).

    For the baseline scenario, callers should set url to point directly at the
    L3 provider endpoint and pass an empty api_key so no X-API-Key header is
    sent.  The layer header will be absent in that case, so we default to 'l3'.
    """
    body = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            layer = resp.headers.get("X-Isartor-Layer", "l3").lower().replace("-", "")
            return layer, latency_ms
    except urllib.error.HTTPError as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        raise RuntimeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------


def run_scenario(
    scenario: str,
    prompts: list[str],
    *,
    url: str,
    api_key: str,
    dry_run: bool,
    timeout: float,
) -> dict:
    """Run one benchmark scenario and return a result dict."""
    counts: dict[str, int] = {"l1a": 0, "l1b": 0, "l2": 0, "l3": 0, "error": 0}
    latencies: list[float] = []
    layer_latencies: dict[str, list[float]] = {k: [] for k in ("l1a", "l1b", "l2", "l3")}

    total = len(prompts)
    print(f"\n{'─' * 60}")
    print(f"  Scenario : {scenario}")
    print(f"  Prompts  : {total}")
    print(f"  Dry-run  : {dry_run}")
    print(f"{'─' * 60}")

    for i, prompt in enumerate(prompts, 1):
        try:
            if dry_run:
                layer, latency_ms = _simulate_response(prompt, scenario)
            else:
                layer, latency_ms = send_request(
                    url, prompt, api_key=api_key, timeout=timeout
                )
            counts[layer] = counts.get(layer, 0) + 1
            latencies.append(latency_ms)
            layer_latencies[layer].append(latency_ms)
        except RuntimeError as exc:
            counts["error"] += 1
            print(f"  [WARN] request {i}/{total} failed: {exc}")

        if i % 25 == 0 or i == total:
            print(f"  Progress: {i}/{total}", end="\r", flush=True)

    print()  # newline after progress

    # ── Compute summary stats ────────────────────────────────────────────
    good_total = total - counts["error"]
    deflected = counts["l1a"] + counts["l1b"] + counts["l2"]
    deflection_rate = deflected / good_total if good_total else 0.0
    error_rate = counts["error"] / total if total else 0.0

    latencies.sort()
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)

    def layer_p50(layer: str) -> float | None:
        lats = sorted(layer_latencies.get(layer, []))
        return _percentile(lats, 50) if lats else None

    tokens_saved = AVG_PROMPT_TOKENS * deflected
    cost_saved_usd = tokens_saved * GPT4O_INPUT_PRICE_PER_TOKEN
    cost_per_req_usd = cost_saved_usd / total if total else 0.0

    result = {
        "scenario": scenario,
        "total_requests": total,
        "l1a_hits": counts["l1a"],
        "l1b_hits": counts["l1b"],
        "l2_hits": counts["l2"],
        "l3_hits": counts["l3"],
        "error_count": counts["error"],
        "l1a_rate": counts["l1a"] / total if total else 0.0,
        "l1b_rate": counts["l1b"] / total if total else 0.0,
        "l2_rate": counts["l2"] / total if total else 0.0,
        "l3_rate": counts["l3"] / total if total else 0.0,
        "deflection_rate": deflection_rate,
        "error_rate": error_rate,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        **{
            f"{lyr}_p50_ms": (round(v, 2) if (v := layer_p50(lyr)) is not None else None)
            for lyr in ("l1a", "l1b", "l2", "l3")
        },
        "tokens_saved": tokens_saved,
        "cost_saved_usd": round(cost_saved_usd, 6),
        "cost_per_req_usd": round(cost_per_req_usd, 6),
    }

    # ── Print human-readable summary ─────────────────────────────────────
    _print_summary(result)

    return result


def _print_summary(r: dict) -> None:
    total = r["total_requests"]
    print()
    print(f"  ── {r['scenario']} ──")
    print(f"  Total requests : {total:5d}")
    print(f"  L1a (exact)    : {r['l1a_hits']:5d}  ({r['l1a_rate'] * 100:.1f}%)")
    print(f"  L1b (semantic) : {r['l1b_hits']:5d}  ({r['l1b_rate'] * 100:.1f}%)")
    print(f"  L2  (SLM)      : {r['l2_hits']:5d}  ({r['l2_rate'] * 100:.1f}%)")
    print(f"  L3  (cloud)    : {r['l3_hits']:5d}  ({r['l3_rate'] * 100:.1f}%)")
    print(f"  Errors         : {r['error_count']:5d}  ({r['error_rate'] * 100:.1f}%)")
    print(f"  Deflection rate: {r['deflection_rate'] * 100:.1f}%")
    print(f"  P50 latency    : {r['p50_ms']:.1f} ms")
    print(f"  P95 latency    : {r['p95_ms']:.1f} ms")
    print(f"  P99 latency    : {r['p99_ms']:.1f} ms")
    print(f"  Cost saved     : ${r['cost_saved_usd']:.4f}  (${r['cost_per_req_usd']:.6f}/req)")


# ---------------------------------------------------------------------------
# Acceptance-criteria check
# ---------------------------------------------------------------------------


def check_acceptance(results: dict[str, dict]) -> bool:
    """
    Evaluate acceptance criteria across all scenarios and print a pass/fail
    report.  Returns True only when every criterion passes.
    """
    print("\n" + "═" * 60)
    print("  ACCEPTANCE CRITERIA")
    print("═" * 60)

    all_pass = True

    def check(label: str, value: float, threshold: float, op: str = ">=") -> bool:
        if op == ">=":
            ok = value >= threshold
        else:
            ok = value < threshold
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {label}: {value * 100:.1f}%  (threshold: {op} {threshold * 100:.0f}%)")
        return ok

    if "warm" in results:
        r = results["warm"]
        all_pass &= check(
            "warm  deflection rate",
            r["deflection_rate"],
            WARM_DEFLECTION_MIN,
        )
        all_pass &= check(
            "warm  error rate     ",
            r["error_rate"],
            MAX_ERROR_RATE,
            op="<",
        )

    if "cold" in results:
        r = results["cold"]
        all_pass &= check(
            "cold  deflection rate",
            r["deflection_rate"],
            COLD_DEFLECTION_MIN,
        )
        all_pass &= check(
            "cold  error rate     ",
            r["error_rate"],
            MAX_ERROR_RATE,
            op="<",
        )

    if "baseline" in results:
        r = results["baseline"]
        all_pass &= check(
            "baseline error rate  ",
            r["error_rate"],
            MAX_ERROR_RATE,
            op="<",
        )

    print("═" * 60)
    outcome = "PASS ✓" if all_pass else "FAIL ✗"
    print(f"  Overall: {outcome}")
    print("═" * 60)

    return all_pass


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


def save_results(
    scenarios: list[str],
    results: dict[str, dict],
    *,
    fixture_path: Path,
    dry_run: bool,
    url: str,
) -> Path:
    """Write results to a timestamped JSON file and update latest.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"claude_code_{ts}.json"

    payload = {
        "benchmark": "claude_code_todo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "isartor_url": url,
        "fixture": str(fixture_path),
        "dry_run": dry_run,
        "hardware": f"{platform.processor() or 'unknown CPU'}, {platform.machine()}",
        "scenarios": results,
        "acceptance": {
            "warm_deflection_min": WARM_DEFLECTION_MIN,
            "cold_deflection_min": COLD_DEFLECTION_MIN,
            "max_error_rate": MAX_ERROR_RATE,
        },
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Results written → {out_path}")

    latest = RESULTS_DIR / "claude_code_latest.json"
    latest.write_text(json.dumps(payload, indent=2))
    print(f"  Latest   updated → {latest}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_SCENARIO_CHOICES = ("baseline", "cold", "warm", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Claude Code + GitHub Copilot Three-Scenario Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ISARTOR_URL", "http://localhost:8080"),
        help="Base URL of the running Isartor instance (default: $ISARTOR_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=os.environ.get("ISARTOR_API_KEY", "changeme"),
        help="X-API-Key header value (default: $ISARTOR_API_KEY or 'changeme')",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_FIXTURE),
        help=f"Path to a JSONL fixture file (default: {DEFAULT_FIXTURE})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="Limit the number of prompts per scenario (0 = all)",
    )
    parser.add_argument(
        "--scenario",
        choices=_SCENARIO_CHOICES,
        default="all",
        help="Which scenario(s) to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path for the results JSON file",
# ── Simulation helpers ─────────────────────────────────────────────────────────

def _stable_rng(prompt: str) -> tuple[random.Random, int]:
    """Return a seeded RNG and a 16-bit fingerprint derived from the prompt."""
    digest = hashlib.md5(prompt.encode(), usedforsecurity=False).digest()
    h = int.from_bytes(digest[:2], "little")
    rng = random.Random(int.from_bytes(digest, "little"))
    return rng, h


def _simulate_case_a(prompt: str) -> tuple[str, float]:
    """
    Simulate Case A (without Isartor): every request goes to L3 cloud.
    Latency is drawn from a realistic cloud-LLM distribution (800–2500 ms
    for code generation tasks, with occasional slow outliers).
    """
    rng, _ = _stable_rng("case-a:" + prompt)
    # 80% of requests: 800–1800 ms  (normal cloud latency for code)
    # 15% of requests: 1800–3500 ms (slower completions / tool calls)
    # 5% of requests:  3500–8000 ms (very slow / retry)
    r = rng.random()
    if r < 0.80:
        latency = rng.uniform(800, 1800)
    elif r < 0.95:
        latency = rng.uniform(1800, 3500)
    else:
        latency = rng.uniform(3500, 8000)
    return "l3", latency


def _simulate_case_b(prompt: str) -> tuple[str, float]:
    """
    Simulate Case B (with Isartor + Qwen L2).  Distribution mirrors the
    expected behaviour for a coding workload with warm/cold cache phases:
      L1a ~35% (exact cache — repeated identical prompts)
      L1b ~20% (semantic cache — paraphrased prompts)
      L2  ~12% (Qwen 2.5 Coder 7B resolves novel but simple code tasks)
      L3  ~33% (cloud — novel complex prompts that need a frontier model)
    """
    rng, h = _stable_rng("case-b:" + prompt)
    if h < 0x599A:          # ~35% -> L1a  (sub-ms exact cache)
        layer = "l1a"
        latency = rng.uniform(0.1, 0.8)
    elif h < 0x8000:        # ~20% -> L1b  (1–8 ms semantic cache)
        layer = "l1b"
        latency = rng.uniform(1.0, 8.0)
    elif h < 0x9EB8:        # ~12% -> L2   (50–350 ms local Qwen inference)
        layer = "l2"
        latency = rng.uniform(50.0, 350.0)
    else:                   # ~33% -> L3   (800–2500 ms cloud)
        layer = "l3"
        latency = rng.uniform(800.0, 2500.0)
    return layer, latency


# ── Percentile helper ─────────────────────────────────────────────────────────

def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = min(math.ceil(len(sorted_data) * pct / 100) - 1, len(sorted_data) - 1)
    return sorted_data[max(idx, 0)]


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_case_a(
    prompts: list[str],
    *,
    dry_run: bool = False,
    direct_url: str = "",
    direct_api_key: str = "",
    azure_url: str = "",
    azure_api_key: str = "",
    azure_deployment: str = "",
    azure_api_version: str = "2024-08-01-preview",
    timeout: float = 120.0,
) -> dict:
    """
    Run Case A — without Isartor.

    In live mode, sends each prompt directly to the cloud LLM API.
    Supports two backends:
      - Azure OpenAI (when ``azure_url`` and ``azure_api_key`` are set)
      - Anthropic Messages API (when ``direct_url`` and ``direct_api_key`` are set)
    Every request is expected to reach the cloud (L3) and no deflection
    header is returned.

    In dry-run mode, simulates realistic cloud-LLM latency without a server.
    """
    all_latencies: list[float] = []
    errors = 0

    use_azure = bool(azure_url and azure_api_key and azure_deployment)

    for prompt in prompts:
        if dry_run:
            _, latency_ms = _simulate_case_a(prompt)
            all_latencies.append(latency_ms)
            continue

        start = time.perf_counter()

        if use_azure:
            # Azure OpenAI chat completions format.
            endpoint = (
                f"{azure_url.rstrip('/')}/openai/deployments/{azure_deployment}"
                f"/chat/completions?api-version={azure_api_version}"
            )
            headers = {
                "Content-Type": "application/json",
                "api-key": azure_api_key,
            }
            body = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": AVG_OUTPUT_TOKENS,
            }).encode()
        else:
            # Anthropic Messages API format.
            endpoint = f"{direct_url.rstrip('/')}/v1/messages"
            headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            if direct_api_key:
                headers["x-api-key"] = direct_api_key
            body = json.dumps({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": AVG_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()

        req = urllib.request.Request(endpoint, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                latency_ms = (time.perf_counter() - start) * 1000
                all_latencies.append(latency_ms)
        except urllib.error.HTTPError as exc:
            errors += 1
            print(f"  [warn] Case A HTTP {exc.code}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [warn] Case A request failed: {exc}", file=sys.stderr)

    total = len(prompts)
    if total == 0:
        return _empty_case_a_result()

    sorted_all = sorted(all_latencies)
    p50 = statistics.median(all_latencies) if all_latencies else 0.0
    p95 = _percentile(sorted_all, 95) if all_latencies else 0.0
    p99 = _percentile(sorted_all, 99) if all_latencies else 0.0

    # In Case A every request hits the cloud.
    l3_hits = total - errors
    cloud_input_tokens = l3_hits * AVG_INPUT_TOKENS
    cloud_output_tokens = l3_hits * AVG_OUTPUT_TOKENS

    # Use the appropriate pricing model depending on the backend.
    if use_azure:
        in_price = GPT4O_MINI_INPUT_PRICE_PER_TOKEN
        out_price = GPT4O_MINI_OUTPUT_PRICE_PER_TOKEN
        backend_label = f"Azure OpenAI ({azure_deployment})"
    else:
        in_price = CLAUDE_INPUT_PRICE_PER_TOKEN
        out_price = CLAUDE_OUTPUT_PRICE_PER_TOKEN
        backend_label = "Anthropic (claude-3-5-sonnet)"

    total_cost_usd = cloud_input_tokens * in_price + cloud_output_tokens * out_price
    cost_per_req = total_cost_usd / total if total else 0.0

    _print_case_a_summary(
        f"Case A — without Isartor [{backend_label}]",
        total, l3_hits, errors, p50, p95, p99, total_cost_usd, cost_per_req,
    )

    return {
        "case": "A",
        "label": "without_isartor",
        "backend": backend_label,
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
        "total_cost_usd": round(total_cost_usd, 6),
        "cost_per_req_usd": round(cost_per_req, 8),
    }


def run_case_b(
    prompts: list[str],
    *,
    dry_run: bool = False,
    isartor_url: str = "http://localhost:8080",
    api_key: str = "changeme",
    azure_l3: bool = False,
    timeout: float = 120.0,
) -> dict:
    """
    Run Case B — with Isartor (Qwen 2.5 Coder 7B as Layer 2).

    In live mode, sends each prompt to ``isartor_url/v1/messages`` and reads
    the X-Isartor-Layer response header to determine which layer resolved it.

    In dry-run mode, simulates the expected layer distribution without a server.
    """
    layer_counts: dict[str, int] = {k: 0 for k in LAYERS}
    layer_latencies: dict[str, list[float]] = {k: [] for k in LAYERS}
    all_latencies: list[float] = []
    errors = 0

    for prompt in prompts:
        if dry_run:
            layer, latency_ms = _simulate_case_b(prompt)
            layer_counts[layer] += 1
            layer_latencies[layer].append(latency_ms)
            all_latencies.append(latency_ms)
            continue

        # Live path: call Isartor's native /api/chat endpoint.
        # The native endpoint reliably returns X-Isartor-Layer on every response.
        start = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        body = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(
            f"{isartor_url.rstrip('/')}/api/chat",
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
                print(f"  [warn] Case B HTTP {exc.code}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [warn] Case B request failed: {exc}", file=sys.stderr)

    total = len(prompts)
    if total == 0:
        return _empty_case_b_result()

    deflected = layer_counts["l1a"] + layer_counts["l1b"] + layer_counts["l2"]
    deflection_rate = deflected / total if total else 0.0

    sorted_all = sorted(all_latencies)
    p50 = statistics.median(all_latencies) if all_latencies else 0.0
    p95 = _percentile(sorted_all, 95) if all_latencies else 0.0
    p99 = _percentile(sorted_all, 99) if all_latencies else 0.0

    def layer_p50_val(layer: str) -> float | None:
        lats = layer_latencies.get(layer, [])
        return round(statistics.median(lats), 2) if lats else None

    # Cloud tokens — only L3 requests consume cloud quota.
    l3_hits = layer_counts["l3"]
    cloud_input_tokens = l3_hits * AVG_INPUT_TOKENS
    cloud_output_tokens = l3_hits * AVG_OUTPUT_TOKENS

    # Use Azure pricing if L3 backend is Azure OpenAI, otherwise Claude pricing.
    if azure_l3:
        in_price = GPT4O_MINI_INPUT_PRICE_PER_TOKEN
        out_price = GPT4O_MINI_OUTPUT_PRICE_PER_TOKEN
    else:
        in_price = CLAUDE_INPUT_PRICE_PER_TOKEN
        out_price = CLAUDE_OUTPUT_PRICE_PER_TOKEN

    total_cost_usd = cloud_input_tokens * in_price + cloud_output_tokens * out_price
    cost_per_req = total_cost_usd / total if total else 0.0

    _print_case_b_summary(
        "Case B — with Isartor (Qwen L2)",
        total, layer_counts, layer_latencies, p50, p95, p99,
        deflection_rate, total_cost_usd, cost_per_req,
    )

    return {
        "case": "B",
        "label": "with_isartor_qwen_l2",
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

def _print_case_a_summary(
    label: str,
    total: int,
    l3_hits: int,
    errors: int,
    p50: float,
    p95: float,
    p99: float,
    total_cost: float,
    cost_per_req: float,
) -> None:
    print(f"\n-- {label} --")
    print(f"  Total requests : {total}")
    print(f"  L3  (cloud)    : {l3_hits:5d}  (100.0%)")
    print(f"  Errors         : {errors:5d}")
    print(f"  Deflection rate: 0.0%  (no local deflection — every request hits cloud)")
    print(f"  P50 latency    : {p50:.1f} ms")
    print(f"  P95 latency    : {p95:.1f} ms")
    print(f"  P99 latency    : {p99:.1f} ms")
    print(f"  Est. cloud cost: ${total_cost:.4f}  (${cost_per_req:.6f}/req)")


def _print_case_b_summary(
    label: str,
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

    print(f"\n-- {label} --")
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
    case_a: dict | None,
    case_b: dict | None,
    *,
    total_prompts: int,
    fixture_name: str = "claude_code_todo_app",
    hardware: str = "unknown",
    timestamp: str = "",
) -> str:
    """Build the full Markdown comparison report."""
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
        "This report compares two execution paths for a deterministic TypeScript todo-app",
        "coding workload that simulates a real Claude Code agent session:",
        "",
        "- **Case A — without Isartor:** every prompt is forwarded directly to the cloud",
        "  LLM provider (GitHub Copilot-backed model). No local deflection occurs.",
        "- **Case B — with Isartor:** prompts route through Isartor's L1 cache → L2 Qwen",
        "  2.5 Coder 7B sidecar → L3 cloud. Deflected requests consume zero cloud quota.",
        "",
    ]

    # ── Side-by-side comparison table ────────────────────────────────────────
    if case_a and case_b:
        total = case_b["total_requests"]
        a_cost = case_a["total_cost_usd"]
        b_cost = case_b["total_cost_usd"]
        cost_reduction = (a_cost - b_cost) / a_cost * 100 if a_cost > 0 else 0.0
        cloud_reqs_saved = total - case_b["l3_hits"]
        tokens_avoided = cloud_reqs_saved * (AVG_INPUT_TOKENS + AVG_OUTPUT_TOKENS)

        lines += [
            "## Comparison",
            "",
            "| Metric                        | Case A (no Isartor) | Case B (with Isartor) | Δ |",
            "|-------------------------------|---------------------|------------------------|---|",
            f"| Total requests                | {case_a['total_requests']} | {case_b['total_requests']} | — |",
            f"| Cloud (L3) requests           | {case_a['l3_hits']} (100%) | {case_b['l3_hits']} ({case_b['l3_hits'] / total * 100:.0f}%) | **−{cloud_reqs_saved} ({case_b['deflection_rate'] * 100:.0f}% avoided)** |",
            f"| Deflection rate               | 0% | {case_b['deflection_rate'] * 100:.1f}% | **+{case_b['deflection_rate'] * 100:.1f}pp** |",
            f"| Est. cloud tokens avoided     | — | {tokens_avoided:,} | **−{tokens_avoided:,}** |",
            f"| Est. cloud cost               | ${a_cost:.4f} | ${b_cost:.4f} | **−{cost_reduction:.1f}%** |",
            f"| Overall P50 latency           | {_ms(case_a.get('p50_ms'))} | {_ms(case_b.get('p50_ms'))} | — |",
            f"| Overall P95 latency           | {_ms(case_a.get('p95_ms'))} | {_ms(case_b.get('p95_ms'))} | — |",
            "",
        ]

    # ── Case A detail ─────────────────────────────────────────────────────────
    if case_a:
        total = case_a["total_requests"]
        lines += [
            "## Case A — Without Isartor",
            "",
            "Every request is forwarded directly to the cloud provider. There is no local",
            "cache or on-device model. All latency is cloud-round-trip latency.",
            "",
            "| Layer              | Hits   | % of Traffic | Avg Latency (p50) |",
            "|--------------------|--------|--------------|-------------------|",
            f"| L1a (exact)        |      0 |        0.0%  |                 - |",
            f"| L1b (semantic)     |      0 |        0.0%  |                 - |",
            f"| L2  (SLM)          |      0 |        0.0%  |                 - |",
            f"| L3  (cloud)        | {case_a['l3_hits']:6d} |      100.0%  | {_ms(case_a.get('p50_ms'))} |",
            f"| **Total deflected**|      0 |       **0%** |                   |",
            f"| **Est. cost**      |        |              | **${case_a['cost_per_req_usd']:.6f}/req** |",
            "",
            f"> Overall latency — P50: {_ms(case_a.get('p50_ms'))} | P95: {_ms(case_a.get('p95_ms'))} | P99: {_ms(case_a.get('p99_ms'))}",
            ">",
            f"> Errors: {case_a['error_count']}",
            "",
        ]

    # ── Case B detail ─────────────────────────────────────────────────────────
    if case_b:
        total = case_b["total_requests"]
        deflected = case_b["l1a_hits"] + case_b["l1b_hits"] + case_b["l2_hits"]
        lines += [
            "## Case B — With Isartor (Qwen 2.5 Coder 7B as Layer 2)",
            "",
            "Requests route through Isartor's deflection stack:",
            "L1a exact cache → L1b semantic cache → L2 Qwen 2.5 Coder 7B (llama.cpp) → L3 cloud.",
            "",
            "| Layer              | Hits   | % of Traffic | Avg Latency (p50) |",
            "|--------------------|--------|--------------|-------------------|",
            f"| L1a (exact)        | {case_b['l1a_hits']:6d} | {case_b['l1a_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(case_b, 'l1a'):>17} |",
            f"| L1b (semantic)     | {case_b['l1b_hits']:6d} | {case_b['l1b_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(case_b, 'l1b'):>17} |",
            f"| L2  (Qwen)         | {case_b['l2_hits']:6d} | {case_b['l2_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(case_b, 'l2'):>17} |",
            f"| L3  (cloud)        | {case_b['l3_hits']:6d} | {case_b['l3_hits'] / total * 100:>10.1f}%  | {_layer_p50_fmt(case_b, 'l3'):>17} |",
            f"| **Total deflected**| **{deflected}** | **{case_b['deflection_rate'] * 100:.1f}%** | |",
            f"| **Est. cost**      |        |              | **${case_b['cost_per_req_usd']:.6f}/req** |",
            "",
            f"> Overall latency — P50: {_ms(case_b.get('p50_ms'))} | P95: {_ms(case_b.get('p95_ms'))} | P99: {_ms(case_b.get('p99_ms'))}",
            ">",
            f"> Errors: {case_b['error_count']}",
            "",
        ]

    # ── ROI section ───────────────────────────────────────────────────────────
    if case_a and case_b:
        a_cost = case_a["total_cost_usd"]
        b_cost = case_b["total_cost_usd"]
        savings = a_cost - b_cost
        roi_pct = savings / a_cost * 100 if a_cost > 0 else 0.0
        cloud_reqs_saved = case_a["l3_hits"] - case_b["l3_hits"]
        tokens_avoided = cloud_reqs_saved * (AVG_INPUT_TOKENS + AVG_OUTPUT_TOKENS)

        lines += [
            "## ROI Analysis",
            "",
            "| Metric                        | Value |",
            "|-------------------------------|-------|",
            f"| Cloud requests avoided        | {cloud_reqs_saved} of {case_a['l3_hits']} ({case_b['deflection_rate'] * 100:.1f}%) |",
            f"| Cloud tokens avoided          | {tokens_avoided:,} |",
            f"| Estimated cost without Isartor| ${a_cost:.4f} |",
            f"| Estimated cost with Isartor   | ${b_cost:.4f} |",
            f"| Estimated savings             | ${savings:.4f} |",
            f"| Cost reduction                | {roi_pct:.1f}% |",
            "",
            "**Interpretation:** For a typical Claude Code session replaying this",
            f"todo-app workload ({case_a['total_requests']} prompts), routing through Isartor with",
            f"Qwen 2.5 Coder 7B as Layer 2 avoids approximately **{roi_pct:.0f}%** of cloud",
            "token spend while keeping median latency low for deflected requests.",
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
        "- **Case A control path:** Claude Code → direct Anthropic/Copilot API.",
        "  If a true direct Claude Code + Copilot path is not available without Isartor,",
        "  a simulated cloud-only baseline is used (100% L3, realistic latency distribution).",
        "- **Case B treatment path:** Claude Code → Isartor `/v1/messages` →",
        "  L1a/L1b cache → L2 Qwen 2.5 Coder 7B (llama.cpp Q4_K_M) → L3 cloud.",
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

def _empty_case_a_result() -> dict:
    return {
        "case": "A", "label": "without_isartor",
        "total_requests": 0, "l1a_hits": 0, "l1b_hits": 0, "l2_hits": 0,
        "l3_hits": 0, "error_count": 0, "deflection_rate": 0.0,
        "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
        "cloud_input_tokens": 0, "cloud_output_tokens": 0,
        "total_cost_usd": 0.0, "cost_per_req_usd": 0.0,
    }


def _empty_case_b_result() -> dict:
    return {
        "case": "B", "label": "with_isartor_qwen_l2",
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
    case_a: dict | None,
    case_b: dict | None,
    report_md: str,
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "claude_code_copilot",
        "fixture": "claude_code_todo_app",
        "hardware": hardware_summary(),
        "layer2_model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
    }
    if case_a:
        payload["case_a"] = case_a
    if case_b:
        payload["case_b"] = case_b

    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path.write_text(report_md)
    print(f"\nJSON results  → {output_path}")
    print(f"Markdown report → {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude Code + GitHub Copilot Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    default_isartor_url = os.environ.get("ISARTOR_URL", "http://localhost:8080")
    default_api_key = os.environ.get("ISARTOR_API_KEY", "changeme")
    default_direct_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    default_direct_key = os.environ.get("ANTHROPIC_API_KEY", "")
    default_azure_url = os.environ.get("AZURE_OPENAI_URL", "")
    default_azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    default_azure_deploy = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    default_azure_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    parser.add_argument(
        "--case",
        choices=["A", "B"],
        help="Run a single case: A (without Isartor) or B (with Isartor)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both Case A and Case B and generate a comparison report",
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
        help="Base URL of the Isartor instance (default: $ISARTOR_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        dest="api_key",
        help="X-API-Key for Isartor (default: $ISARTOR_API_KEY or 'changeme')",
    )
    parser.add_argument(
        "--direct-url",
        default=default_direct_url,
        dest="direct_url",
        help="Direct Anthropic API base URL for Case A (default: $ANTHROPIC_BASE_URL)",
    )
    parser.add_argument(
        "--direct-api-key",
        default=default_direct_key,
        dest="direct_api_key",
        help="Anthropic API key for Case A (default: $ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--azure-url",
        default=default_azure_url,
        dest="azure_url",
        help="Azure OpenAI resource URL for Case A (default: $AZURE_OPENAI_URL)",
    )
    parser.add_argument(
        "--azure-api-key",
        default=default_azure_key,
        dest="azure_api_key",
        help="Azure OpenAI API key for Case A (default: $AZURE_OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--azure-deployment",
        default=default_azure_deploy,
        dest="azure_deployment",
        help="Azure OpenAI deployment name (default: $AZURE_OPENAI_DEPLOYMENT or gpt-4o-mini)",
    )
    parser.add_argument(
        "--azure-api-version",
        default=default_azure_version,
        dest="azure_api_version",
        help="Azure OpenAI API version (default: 2024-08-01-preview)",
    )
    parser.add_argument(
        "--input",
        default=str(FIXTURE_FILE),
        help=f"Path to a JSONL fixture file (default: {FIXTURE_FILE})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="Limit number of prompts (0 = all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ISARTOR_TIMEOUT", "120")),
        help="Per-request timeout in seconds (default: $ISARTOR_TIMEOUT or 120)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Simulate responses locally — no server required. "
            "Useful for CI validation and smoke-testing the harness."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Load fixture ─────────────────────────────────────────────────────
    fixture_path = Path(args.input)
    if not fixture_path.exists():
        print(f"[ERROR] Fixture file not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    all_prompts = load_prompts(fixture_path)
    if args.requests > 0:
        all_prompts = all_prompts[: args.requests]

    if not all_prompts:
        print("[ERROR] No prompts loaded from fixture.", file=sys.stderr)
        sys.exit(1)

    # ── Determine which scenarios to run ─────────────────────────────────
    if args.scenario == "all":
        scenarios = ["baseline", "cold", "warm"]
    else:
        scenarios = [args.scenario]

    # ── Banner ────────────────────────────────────────────────────────────
    print("═" * 60)
    print("  Claude Code + GitHub Copilot Benchmark")
    print("═" * 60)
    print(f"  Fixture  : {fixture_path.name}  ({len(all_prompts)} prompts)")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  URL      : {args.url}")
    print(f"  Dry-run  : {args.dry_run}")
    print(f"  Timeout  : {args.timeout}s")
    print("═" * 60)

    # ── Run scenarios ─────────────────────────────────────────────────────
    results: dict[str, dict] = {}

    for scenario in scenarios:
        # For the warm scenario we run the same prompts a second time so the
        # cache is already warm from the cold run.
        results[scenario] = run_scenario(
            scenario,
            all_prompts,
            url=args.url,
            api_key=args.api_key,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )

    # ── Acceptance check ─────────────────────────────────────────────────
    accepted = check_acceptance(results)

    # ── Save results ──────────────────────────────────────────────────────
    save_results(
        scenarios,
        results,
        fixture_path=fixture_path,
        dry_run=args.dry_run,
        url=args.url,
    )

    # ── Exit code ─────────────────────────────────────────────────────────
    sys.exit(0 if accepted else 1)
        "--output",
        default=str(RESULTS_DIR / "claude_code_copilot.json"),
        help="Path for the JSON results file",
    )
    parser.add_argument(
        "--report",
        default=str(RESULTS_DIR / "claude_code_copilot_report.md"),
        help="Path for the Markdown report file",
    )
    args = parser.parse_args()

    if not args.case and not args.compare and not args.dry_run:
        parser.print_help()
        print(
            "\nError: specify --case A, --case B, --compare, or --dry-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --dry-run without explicit --case or --compare implies --compare
    run_a = args.compare or args.case == "A" or args.dry_run
    run_b = args.compare or args.case == "B" or args.dry_run

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

    case_a_result: dict | None = None
    case_b_result: dict | None = None

    if run_a:
        case_a_result = run_case_a(
            prompts,
            dry_run=args.dry_run,
            direct_url=args.direct_url,
            direct_api_key=args.direct_api_key,
            azure_url=args.azure_url,
            azure_api_key=args.azure_api_key,
            azure_deployment=args.azure_deployment,
            azure_api_version=args.azure_api_version,
            timeout=args.timeout,
        )

    if run_b:
        case_b_result = run_case_b(
            prompts,
            dry_run=args.dry_run,
            isartor_url=args.isartor_url,
            api_key=args.api_key,
            azure_l3=bool(args.azure_url and args.azure_api_key),
            timeout=args.timeout,
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hw = hardware_summary()
    report_md = build_markdown_report(
        case_a_result,
        case_b_result,
        total_prompts=len(prompts),
        fixture_name=input_path.stem,
        hardware=hw,
        timestamp=ts,
    )

    print("\n" + "=" * 72)
    print(report_md)

    write_results(
        case_a_result,
        case_b_result,
        report_md,
        Path(args.output),
        Path(args.report),
    )


if __name__ == "__main__":
    main()

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


if __name__ == "__main__":
    main()

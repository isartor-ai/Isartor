#!/usr/bin/env python3
"""
Claude Code TypeScript Todo-App Benchmark Harness
==================================================

Deterministic benchmark that replays a fixed set of Claude Code prompts for
building a TypeScript todo app through three scenarios:

  baseline  – sends prompts directly to the LLM with Isartor bypassed
               (or simulated in --dry-run mode); control path.
  cold      – sends prompts through Isartor with an empty cache (first run).
  warm      – replays the identical prompts through Isartor after the cold run
               has already seeded the cache; measures L1a (exact cache) hit rate.

Usage
-----
  # All three scenarios (dry-run, no server required):
  python3 benchmarks/claude_code_benchmark.py --dry-run

  # Single scenario against a live Isartor instance:
  python3 benchmarks/claude_code_benchmark.py --scenario cold \\
      --url http://localhost:8080 --api-key changeme

  # All scenarios live, post results to a GitHub issue:
  GH_TOKEN=ghp_xxx \\
  python3 benchmarks/claude_code_benchmark.py --all-scenarios \\
      --url http://localhost:8080 --api-key changeme \\
      --repo isartor-ai/Isartor --issue 42

  # Make shortcut:
  make benchmark-claude-code-dry-run
  make benchmark-claude-code

Environment variables (can substitute for CLI flags)
-----------------------------------------------------
  ISARTOR_URL          base URL of the Isartor gateway (default: http://localhost:8080)
  ISARTOR_API_KEY      X-API-Key value (default: changeme)
  GH_TOKEN             GitHub personal access token for issue comments
  GH_REPO              owner/repo (e.g. isartor-ai/Isartor)
  GH_ISSUE             issue number to post progress comments to

Output
------
  benchmarks/results/claude_code_<scenario>_<timestamp>.json   per-scenario result
  benchmarks/results/claude_code_summary.json                  aggregated summary
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
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
FIXTURE_PATH = SCRIPT_DIR / "fixtures" / "claude_code_todo_app.jsonl"
RESULTS_DIR = SCRIPT_DIR / "results"

# ── Constants ─────────────────────────────────────────────────────────────────
# Claude Sonnet 3.5 input-token price (USD / token) — used for cost estimates.
CLAUDE_INPUT_PRICE_PER_TOKEN = 0.000003
# Conservative average token count per todo-app prompt.
AVG_PROMPT_TOKENS = 120

LAYERS = ("l1a", "l1b", "l2", "l3")
SCENARIOS = ("baseline", "cold", "warm")


# ── Fixture loading ────────────────────────────────────────────────────────────

def load_task_packet(path: Path) -> list[dict]:
    """Load the task packet from a JSONL file.

    Each line must be a JSON object with at least a ``prompt`` key.
    Optional metadata fields (``phase``, ``step``) are preserved for reporting.
    """
    tasks: list[dict] = []
    with path.open() as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] {path}:{lineno}: JSON parse error: {exc}",
                    file=sys.stderr,
                )
                continue
            if "prompt" not in obj:
                print(
                    f"[warn] {path}:{lineno}: missing 'prompt' key; skipping.",
                    file=sys.stderr,
                )
                continue
            tasks.append(obj)
    return tasks


# ── Dry-run simulation ─────────────────────────────────────────────────────────

def _simulate_response(prompt: str, scenario: str) -> tuple[str, float]:
    """Return a deterministic (layer, latency_ms) pair for dry-run mode.

    Distributions are calibrated to be realistic per scenario:
      baseline  – all L3 (no cache).
      cold      – mostly L3 with a few L2 hits; cache not yet warm.
      warm      – heavy L1a exact hits because the same prompts were seen in cold.
    """
    digest = hashlib.md5(prompt.encode(), usedforsecurity=False).digest()
    seed = int.from_bytes(digest, "little")
    h = int.from_bytes(digest[:2], "little")
    rng = random.Random(seed)

    if scenario == "baseline":
        # Control path: everything goes to cloud.
        return "l3", rng.uniform(400.0, 1200.0)

    if scenario == "cold":
        # First pass through Isartor: mostly L3, some L2.
        if h < 0x0800:            # ~3 %  L2
            return "l2", rng.uniform(50.0, 200.0)
        return "l3", rng.uniform(400.0, 1200.0)    # ~97 % L3

    # warm: same prompts => L1a exact hits dominate
    if h < 0xD000:                # ~82 % L1a
        return "l1a", rng.uniform(0.1, 0.8)
    if h < 0xE800:                # ~9 %  L1b
        return "l1b", rng.uniform(1.0, 5.0)
    if h < 0xF000:                # ~3 %  L2
        return "l2", rng.uniform(50.0, 200.0)
    return "l3", rng.uniform(400.0, 1200.0)        # ~6 %  L3


# ── Isartor snapshot ───────────────────────────────────────────────────────────

def fetch_isartor_snapshot(url: str, api_key: str) -> dict[str, Any]:
    """Fetch current Isartor stats from the /health or /debug/stats endpoint."""
    snapshot: dict[str, Any] = {}

    def _get(path: str) -> dict | None:
        headers = {"X-API-Key": api_key} if api_key else {}
        req = urllib.request.Request(
            f"{url}{path}",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            return None
        except json.JSONDecodeError:
            return None

    health = _get("/health")
    if health:
        snapshot["health"] = health

    stats = _get("/debug/stats")
    if stats:
        snapshot["stats"] = stats

    snapshot["captured_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return snapshot


# ── Core benchmark engine ──────────────────────────────────────────────────────

def run_scenario(
    tasks: list[dict],
    scenario: str,
    *,
    url: str,
    api_key: str,
    dry_run: bool = False,
    timeout: float = 120.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one scenario and return a result dict.

    Parameters
    ----------
    tasks:    List of task objects loaded from the fixture file.
    scenario: One of 'baseline', 'cold', 'warm'.
    url:      Base URL of the Isartor gateway.
    api_key:  X-API-Key value.
    dry_run:  When True, simulate responses without a live server.
    timeout:  Per-request HTTP timeout (seconds).
    verbose:  Print per-request progress lines.
    """
    counts: dict[str, int] = {"l1a": 0, "l1b": 0, "l2": 0, "l3": 0, "error": 0}
    layer_latencies: dict[str, list[float]] = {k: [] for k in LAYERS}
    all_latencies: list[float] = []
    per_task_results: list[dict] = []

    phase_counts: dict[str, dict[str, int]] = {}

    for task in tasks:
        prompt = task["prompt"]
        phase = task.get("phase", "unknown")
        step = task.get("step", 0)

        if phase not in phase_counts:
            phase_counts[phase] = {k: 0 for k in [*LAYERS, "error"]}

        if dry_run:
            layer, latency_ms = _simulate_response(prompt, scenario)
        else:
            layer, latency_ms = _send_request(
                url=url,
                prompt=prompt,
                api_key=api_key,
                timeout=timeout,
            )

        if layer == "error":
            counts["error"] += 1
            phase_counts[phase]["error"] += 1
            per_task_results.append(
                {"phase": phase, "step": step, "layer": "error", "latency_ms": None}
            )
            if verbose:
                print(f"  step {step:2d} [{phase:8s}] ERROR")
            continue

        counts[layer] = counts.get(layer, 0) + 1
        phase_counts[phase][layer] = phase_counts[phase].get(layer, 0) + 1
        layer_latencies[layer].append(latency_ms)
        all_latencies.append(latency_ms)
        per_task_results.append(
            {"phase": phase, "step": step, "layer": layer, "latency_ms": round(latency_ms, 2)}
        )

        if verbose:
            print(
                f"  step {step:2d} [{phase:8s}] layer={layer}  "
                f"latency={latency_ms:8.1f} ms"
            )

    total = len(tasks)
    deflected = counts["l1a"] + counts["l1b"] + counts["l2"]
    deflection_rate = deflected / total if total else 0.0

    sorted_all = sorted(all_latencies)
    p50 = statistics.median(all_latencies) if all_latencies else 0.0
    p95 = _percentile(sorted_all, 95) if all_latencies else 0.0
    p99 = _percentile(sorted_all, 99) if all_latencies else 0.0

    def layer_p50(layer: str) -> float | None:
        lats = layer_latencies.get(layer, [])
        return round(statistics.median(lats), 2) if lats else None

    tokens_saved = AVG_PROMPT_TOKENS * deflected
    cost_saved = tokens_saved * CLAUDE_INPUT_PRICE_PER_TOKEN
    cost_per_req = cost_saved / total if total else 0.0

    return {
        "scenario": scenario,
        "total_tasks": total,
        "deflection_rate": round(deflection_rate, 4),
        "l1a_hits": counts["l1a"],
        "l1b_hits": counts["l1b"],
        "l2_hits": counts["l2"],
        "l3_hits": counts["l3"],
        "l1a_rate": round(counts["l1a"] / total, 4) if total else 0.0,
        "l1b_rate": round(counts["l1b"] / total, 4) if total else 0.0,
        "l2_rate": round(counts["l2"] / total, 4) if total else 0.0,
        "l3_rate": round(counts["l3"] / total, 4) if total else 0.0,
        "error_count": counts["error"],
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "l1a_p50_ms": layer_p50("l1a"),
        "l1b_p50_ms": layer_p50("l1b"),
        "l2_p50_ms": layer_p50("l2"),
        "l3_p50_ms": layer_p50("l3"),
        "tokens_saved": tokens_saved,
        "cost_saved_usd": round(cost_saved, 6),
        "cost_per_req_usd": round(cost_per_req, 8),
        "phase_breakdown": phase_counts,
        "per_task_results": per_task_results,
    }


def _send_request(
    *,
    url: str,
    prompt: str,
    api_key: str,
    timeout: float,
) -> tuple[str, float]:
    """Send one prompt to the Isartor /api/chat endpoint.

    Returns (layer, latency_ms) where layer is one of LAYERS, or 'error'.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=json.dumps({"prompt": prompt}).encode(),
        headers=headers,
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_layer = resp.headers.get("X-Isartor-Layer", "l3")
            latency_ms = (time.perf_counter() - start) * 1000
            if raw_layer not in LAYERS:
                print(
                    f"  [warn] unexpected X-Isartor-Layer: {raw_layer!r}; "
                    "counting as error.",
                    file=sys.stderr,
                )
                return "error", latency_ms
            return raw_layer, latency_ms
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        if exc.code == 401:
            print(
                "  [warn] 401 Unauthorized — check --api-key / $ISARTOR_API_KEY",
                file=sys.stderr,
            )
        else:
            print(f"  [warn] HTTP {exc.code}: {exc}", file=sys.stderr)
        return "error", latency_ms
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"  [warn] request failed: {exc}", file=sys.stderr)
        return "error", latency_ms


# ── Pretty printing ────────────────────────────────────────────────────────────

def print_scenario_summary(result: dict) -> None:
    """Print a human-readable console summary for one scenario."""
    scenario = result["scenario"]
    total = result["total_tasks"]
    print(f"\n{'─' * 60}")
    print(f"  Scenario : {scenario.upper()}")
    print(f"  Tasks    : {total}")
    print(f"  L1a (exact)    : {result['l1a_hits']:5d}  ({result['l1a_rate'] * 100:.1f}%)")
    print(f"  L1b (semantic) : {result['l1b_hits']:5d}  ({result['l1b_rate'] * 100:.1f}%)")
    print(f"  L2  (SLM)      : {result['l2_hits']:5d}  ({result['l2_rate'] * 100:.1f}%)")
    print(f"  L3  (cloud)    : {result['l3_hits']:5d}  ({result['l3_rate'] * 100:.1f}%)")
    print(f"  Errors         : {result['error_count']:5d}")
    print(f"  Deflection     : {result['deflection_rate'] * 100:.1f}%")
    print(f"  P50 latency    : {result['p50_ms']:.1f} ms")
    print(f"  P95 latency    : {result['p95_ms']:.1f} ms")
    print(f"  P99 latency    : {result['p99_ms']:.1f} ms")
    print(f"  Cost saved     : ${result['cost_saved_usd']:.4f}")
    print(f"  Phase breakdown:")
    for phase, counts in result.get("phase_breakdown", {}).items():
        deflected_in_phase = counts.get("l1a", 0) + counts.get("l1b", 0) + counts.get("l2", 0)
        total_in_phase = sum(
            counts.get(k, 0) for k in [*LAYERS, "error"]
        )
        print(
            f"    {phase:10s}: {total_in_phase} tasks, "
            f"{deflected_in_phase} deflected"
        )


def build_markdown_report(results: list[dict], dry_run: bool = False) -> str:
    """Build a Markdown report for all scenarios (suitable for a GitHub comment)."""
    lines: list[str] = [
        "## Claude Code TypeScript Todo-App Benchmark Results",
        "",
    ]
    if dry_run:
        lines += [
            "> ⚠️ **Dry-run mode** — responses were simulated locally; "
            "no live Isartor instance was used.",
            "",
        ]

    for result in results:
        scenario = result["scenario"]
        total = result["total_tasks"]
        defl_pct = result["deflection_rate"] * 100

        lines += [
            f"### Scenario: `{scenario}`",
            "",
            "| Layer | Hits | % | P50 Latency |",
            "|-------|------|---|-------------|",
            f"| L1a (exact)    | {result['l1a_hits']} | {result['l1a_rate']*100:.1f}% "
            f"| {_fmt_ms(result['l1a_p50_ms'])} |",
            f"| L1b (semantic) | {result['l1b_hits']} | {result['l1b_rate']*100:.1f}% "
            f"| {_fmt_ms(result['l1b_p50_ms'])} |",
            f"| L2  (SLM)      | {result['l2_hits']} | {result['l2_rate']*100:.1f}% "
            f"| {_fmt_ms(result['l2_p50_ms'])} |",
            f"| L3  (cloud)    | {result['l3_hits']} | {result['l3_rate']*100:.1f}% "
            f"| {_fmt_ms(result['l3_p50_ms'])} |",
            f"| **Deflected**  | **{result['l1a_hits']+result['l1b_hits']+result['l2_hits']}** "
            f"| **{defl_pct:.1f}%** | |",
            "",
            f"> {total} tasks · P50 {result['p50_ms']:.1f} ms · "
            f"P95 {result['p95_ms']:.1f} ms · P99 {result['p99_ms']:.1f} ms · "
            f"Cost saved ${result['cost_saved_usd']:.4f}",
            "",
        ]

    # Warm-vs-cold comparison table if both present
    cold = next((r for r in results if r["scenario"] == "cold"), None)
    warm = next((r for r in results if r["scenario"] == "warm"), None)
    if cold and warm:
        if warm["p50_ms"] > 0:
            speedup = f"{cold['p50_ms'] / warm['p50_ms']:.0f}×"
        elif cold["p50_ms"] > 0:
            speedup = "∞× (warm P50 rounds to 0)"
        else:
            speedup = "N/A"
        lines += [
            "### Cold vs Warm Cache Comparison",
            "",
            "| Metric | Cold | Warm | Δ |",
            "|--------|------|------|---|",
            f"| Deflection rate | {cold['deflection_rate']*100:.1f}% "
            f"| {warm['deflection_rate']*100:.1f}% "
            f"| +{(warm['deflection_rate']-cold['deflection_rate'])*100:.1f}pp |",
            f"| P50 latency | {cold['p50_ms']:.1f} ms "
            f"| {warm['p50_ms']:.1f} ms "
            f"| warm is {speedup} faster |",
            f"| L1a hits | {cold['l1a_hits']} "
            f"| {warm['l1a_hits']} "
            f"| +{warm['l1a_hits'] - cold['l1a_hits']} |",
            "",
        ]

    return "\n".join(lines)


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.1f} ms"


def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = min(math.ceil(len(sorted_data) * pct / 100) - 1, len(sorted_data) - 1)
    return sorted_data[max(idx, 0)]


# ── GitHub issue reporting ─────────────────────────────────────────────────────

def post_github_comment(
    repo: str,
    issue_number: int,
    body: str,
    token: str,
) -> bool:
    """Post a comment to a GitHub issue.  Returns True on success."""
    owner, _, repo_name = repo.partition("/")
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo_name}"
        f"/issues/{issue_number}/comments"
    )
    payload = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status in (200, 201):
                print(
                    f"  [ok] Posted results to GitHub {repo}#{issue_number}",
                    file=sys.stderr,
                )
                return True
            print(
                f"  [warn] Unexpected GitHub API status {resp.status}",
                file=sys.stderr,
            )
            return False
    except urllib.error.HTTPError as exc:
        print(
            f"  [warn] GitHub API error {exc.code}: {exc.reason}",
            file=sys.stderr,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] Failed to post GitHub comment: {exc}", file=sys.stderr)
        return False


# ── Results persistence ────────────────────────────────────────────────────────

def _hardware_summary() -> str:
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
        return "unknown"


def write_scenario_result(result: dict, scenario: str) -> Path:
    """Persist a single scenario result to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"claude_code_{scenario}_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "hardware": _hardware_summary(),
                "fixture": "claude_code_todo_app.jsonl",
                "result": result,
            },
            indent=2,
        )
        + "\n"
    )
    return out_path


def write_summary(results: list[dict]) -> Path:
    """Persist aggregated summary for all scenarios."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "claude_code_summary.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": _hardware_summary(),
        "fixture": "claude_code_todo_app.jsonl",
        "scenarios": {r["scenario"]: r for r in results},
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Claude Code TypeScript Todo-App Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help=(
            "Run a single scenario (baseline | cold | warm). "
            "Use --all-scenarios to run all three."
        ),
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        dest="all_scenarios",
        help="Run baseline, cold, and warm scenarios in order.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ISARTOR_URL", "http://localhost:8080"),
        help="Base URL of the Isartor gateway (default: $ISARTOR_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ISARTOR_API_KEY", "changeme"),
        dest="api_key",
        help="X-API-Key value (default: $ISARTOR_API_KEY or changeme)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Simulate responses locally — no server required (useful for CI).",
    )
    parser.add_argument(
        "--fixture",
        default=str(FIXTURE_PATH),
        help=f"Path to task-packet JSONL file (default: {FIXTURE_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ISARTOR_TIMEOUT", "120")),
        help="Per-request timeout in seconds (default: $ISARTOR_TIMEOUT or 120)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-task progress lines.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Capture Isartor /health and /debug/stats snapshots before and after each run.",
    )
    # GitHub issue reporting
    parser.add_argument(
        "--repo",
        default=os.environ.get("GH_REPO", ""),
        help="GitHub repo (owner/repo) to post results to (default: $GH_REPO)",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=int(os.environ.get("GH_ISSUE", "0")),
        help="GitHub issue number to comment on (default: $GH_ISSUE)",
    )
    parser.add_argument(
        "--gh-token",
        default=os.environ.get("GH_TOKEN", ""),
        dest="gh_token",
        help="GitHub personal access token (default: $GH_TOKEN)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.scenario and not args.all_scenarios and not args.dry_run:
        # Default: dry-run all scenarios when no explicit selection given.
        parser.print_help()
        print(
            "\nHint: use --all-scenarios [--dry-run] or --scenario <name>.",
            file=sys.stderr,
        )
        sys.exit(1)

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"Error: fixture not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    tasks = load_task_packet(fixture_path)
    if not tasks:
        print("Error: fixture loaded 0 tasks.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(tasks)} tasks from {fixture_path.name}")
    phases = sorted({t.get("phase", "?") for t in tasks})
    print(f"Phases: {', '.join(phases)}")

    scenarios_to_run: list[str]
    if args.all_scenarios:
        scenarios_to_run = list(SCENARIOS)
    elif args.scenario:
        scenarios_to_run = [args.scenario]
    else:
        # --dry-run without --scenario implies all scenarios
        scenarios_to_run = list(SCENARIOS)

    all_results: list[dict] = []

    for scenario in scenarios_to_run:
        print(f"\n{'═' * 60}")
        print(f"  Running scenario: {scenario.upper()}")
        print(f"{'═' * 60}")

        # Optional snapshot before
        snapshot_before: dict | None = None
        if args.snapshot and not args.dry_run:
            snapshot_before = fetch_isartor_snapshot(args.url, args.api_key)
            print(
                f"  [snapshot] captured before {scenario} run "
                f"at {snapshot_before.get('captured_at', '?')}"
            )

        result = run_scenario(
            tasks=tasks,
            scenario=scenario,
            url=args.url,
            api_key=args.api_key,
            dry_run=args.dry_run,
            timeout=args.timeout,
            verbose=args.verbose,
        )

        # Optional snapshot after
        if args.snapshot and not args.dry_run:
            snapshot_after = fetch_isartor_snapshot(args.url, args.api_key)
            result["isartor_snapshot_before"] = snapshot_before
            result["isartor_snapshot_after"] = snapshot_after

        print_scenario_summary(result)

        out_path = write_scenario_result(result, scenario)
        print(f"\n  Results saved to {out_path}")

        all_results.append(result)

    # Write aggregated summary
    summary_path = write_summary(all_results)
    print(f"\nAggregated summary saved to {summary_path}")

    # Build Markdown report
    md_report = build_markdown_report(all_results, dry_run=args.dry_run)
    print("\n" + "─" * 60)
    print("Markdown report (copy-pasteable):")
    print("─" * 60)
    print(md_report)

    # Post to GitHub issue if configured
    if args.repo and args.issue and args.gh_token:
        print(f"\nPosting results to GitHub {args.repo}#{args.issue} …")
        post_github_comment(
            repo=args.repo,
            issue_number=args.issue,
            body=md_report,
            token=args.gh_token,
        )
    elif args.repo or args.issue or args.gh_token:
        print(
            "\n[warn] Incomplete GitHub config — need --repo, --issue, and --gh-token "
            "(or $GH_REPO, $GH_ISSUE, $GH_TOKEN) to post a comment.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

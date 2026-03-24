#!/usr/bin/env python3
"""
Isartor — Claude Code Benchmark ROI Report Generator

Reads the machine-readable benchmark results produced by claude_code_benchmark.py
and emits:
  1. A human-readable Markdown report (benchmarks/results/claude_code_roi_report.md)
  2. An updated machine-readable artifact with ROI fields appended
     (benchmarks/results/claude_code_roi_<timestamp>.json)

Usage
-----
  # Generate from the latest run:
  python3 benchmarks/roi_report.py

  # Generate from a specific results file:
  python3 benchmarks/roi_report.py \
      --input benchmarks/results/claude_code_latest.json

  # Override output paths:
  python3 benchmarks/roi_report.py \
      --input  benchmarks/results/claude_code_latest.json \
      --output benchmarks/results/my_report.md

  # Dry-run (print to stdout, don't write files):
  python3 benchmarks/roi_report.py --print-only

Assumptions
-----------
  All assumptions are documented explicitly in the generated report.

  Token pricing:
    - gpt-4o input:  $0.000005 / token  (OpenAI public pricing, benchmark baseline)
    - gpt-4o output: $0.000015 / token  (OpenAI public pricing, benchmark baseline)

  Token volume:
    - avg_input_tokens:  75 tokens / request
      (code prompts are larger than FAQ prompts; conservative estimate)
    - avg_output_tokens: 300 tokens / request
      (typical Claude Code response for code generation)

  Cloud cost saved formula:
    tokens_deflected = avg_input_tokens  × deflected_requests
    output_saved     = avg_output_tokens × deflected_requests
    cost_saved       = (tokens_deflected × input_price) + (output_saved × output_price)

  ROI formula (simplified payback):
    roi_multiple = cost_saved_per_month / isartor_self_hosting_cost_per_month
    isartor_self_hosting_cost_per_month = $50 USD  (small VM + memory)

  Latency improvement:
    p50_reduction_pct = (baseline_p50 - warm_p50) / baseline_p50 × 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARKS_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
DEFAULT_INPUT = RESULTS_DIR / "claude_code_latest.json"
DEFAULT_OUTPUT_MD = RESULTS_DIR / "claude_code_roi_report.md"

# ---------------------------------------------------------------------------
# Cost / ROI constants  (all assumptions documented in the report)
# ---------------------------------------------------------------------------

GPT4O_INPUT_PRICE_PER_TOKEN = 0.000005    # USD
GPT4O_OUTPUT_PRICE_PER_TOKEN = 0.000015   # USD
AVG_INPUT_TOKENS = 75                     # tokens per request
AVG_OUTPUT_TOKENS = 300                   # tokens per response
ISARTOR_HOSTING_COST_PER_MONTH = 50.0    # USD — small VM estimate
REQUESTS_PER_MONTH_DEFAULT = 50_000       # workday sessions × requests/session

# ---------------------------------------------------------------------------
# ROI computation
# ---------------------------------------------------------------------------


def compute_roi(scenarios: dict[str, dict], requests_per_month: int) -> dict:
    """Compute ROI figures from benchmark scenario results."""

    def _get(scenario: str, key: str, default: float = 0.0) -> float:
        return scenarios.get(scenario, {}).get(key, default)

    baseline_p50 = _get("baseline", "p50_ms")
    if baseline_p50 == 0.0:
        print(
            "[roi_report] WARNING: baseline p50_ms is zero or missing — "
            "latency reduction figures will be inaccurate.",
            file=sys.stderr,
        )
        baseline_p50 = 1.0  # prevent division by zero; already warned above
    warm_p50 = _get("warm", "p50_ms")
    cold_deflection = _get("cold", "deflection_rate")
    warm_deflection = _get("warm", "deflection_rate")

    # ── Per-request cost without Isartor ──
    cost_per_req_no_isartor = (
        AVG_INPUT_TOKENS * GPT4O_INPUT_PRICE_PER_TOKEN
        + AVG_OUTPUT_TOKENS * GPT4O_OUTPUT_PRICE_PER_TOKEN
    )

    # ── Per-request cost with Isartor (warm, steady-state) ──
    # Deflected requests cost ≈ $0 cloud token spend.
    # Non-deflected requests still incur full cloud cost.
    cost_per_req_with_isartor = cost_per_req_no_isartor * (1.0 - warm_deflection)

    # ── Monthly cloud cost delta ──
    monthly_cloud_without = cost_per_req_no_isartor * requests_per_month
    monthly_cloud_with = cost_per_req_with_isartor * requests_per_month
    monthly_cloud_saved = monthly_cloud_without - monthly_cloud_with

    # ── Net monthly benefit (savings minus self-hosting cost) ──
    net_monthly_benefit = monthly_cloud_saved - ISARTOR_HOSTING_COST_PER_MONTH

    # ── ROI multiple ──
    roi_multiple = (
        monthly_cloud_saved / ISARTOR_HOSTING_COST_PER_MONTH
        if ISARTOR_HOSTING_COST_PER_MONTH > 0
        else float("inf")
    )

    # ── Latency improvement ──
    p50_reduction_ms = baseline_p50 - warm_p50
    p50_reduction_pct = (p50_reduction_ms / baseline_p50 * 100) if baseline_p50 > 0 else 0.0

    # ── Token counts ──
    warm_total = _get("warm", "total_requests", 1)
    warm_deflected = int(warm_total * warm_deflection)
    tokens_saved_per_run = (
        warm_deflected * AVG_INPUT_TOKENS
        + warm_deflected * AVG_OUTPUT_TOKENS
    )
    if warm_total > 0:
        scaling_factor = requests_per_month / warm_total
        tokens_saved_per_month = int(tokens_saved_per_run * scaling_factor)
    else:
        tokens_saved_per_month = 0

    return {
        "assumptions": {
            "gpt4o_input_price_per_token_usd": GPT4O_INPUT_PRICE_PER_TOKEN,
            "gpt4o_output_price_per_token_usd": GPT4O_OUTPUT_PRICE_PER_TOKEN,
            "avg_input_tokens_per_request": AVG_INPUT_TOKENS,
            "avg_output_tokens_per_request": AVG_OUTPUT_TOKENS,
            "isartor_hosting_cost_per_month_usd": ISARTOR_HOSTING_COST_PER_MONTH,
            "requests_per_month": requests_per_month,
        },
        "layer_breakdown": {
            "baseline_l3_rate": _get("baseline", "l3_rate"),
            "cold_deflection_rate": cold_deflection,
            "warm_deflection_rate": warm_deflection,
            "warm_l1a_rate": _get("warm", "l1a_rate"),
            "warm_l1b_rate": _get("warm", "l1b_rate"),
            "warm_l2_rate": _get("warm", "l2_rate"),
            "warm_l3_rate": _get("warm", "l3_rate"),
        },
        "cost": {
            "cost_per_req_no_isartor_usd": round(cost_per_req_no_isartor, 8),
            "cost_per_req_with_isartor_usd": round(cost_per_req_with_isartor, 8),
            "monthly_cloud_cost_without_isartor_usd": round(monthly_cloud_without, 2),
            "monthly_cloud_cost_with_isartor_usd": round(monthly_cloud_with, 2),
            "monthly_cloud_saved_usd": round(monthly_cloud_saved, 2),
            "monthly_isartor_hosting_cost_usd": ISARTOR_HOSTING_COST_PER_MONTH,
            "net_monthly_benefit_usd": round(net_monthly_benefit, 2),
        },
        "roi": {
            "roi_multiple": round(roi_multiple, 1),
            "tokens_saved_per_run": tokens_saved_per_run,
            "tokens_saved_per_month_est": tokens_saved_per_month,
        },
        "latency": {
            "baseline_p50_ms": baseline_p50,
            "warm_p50_ms": warm_p50,
            "p50_reduction_ms": round(p50_reduction_ms, 2),
            "p50_reduction_pct": round(p50_reduction_pct, 1),
        },
    }


# ---------------------------------------------------------------------------
# Markdown report renderer
# ---------------------------------------------------------------------------


def render_markdown(raw: dict, roi: dict) -> str:
    """Render the full Markdown benchmark report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scenarios = raw.get("scenarios", {})
    a = roi["assumptions"]
    cost = roi["cost"]
    lb = roi["layer_breakdown"]
    latency = roi["latency"]
    roi_vals = roi["roi"]

    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def ms(v: float | None) -> str:
        return f"{v:.1f} ms" if v is not None else "—"

    def usd(v: float) -> str:
        return f"${v:,.4f}"

    def usd2(v: float) -> str:
        return f"${v:,.2f}"

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────
    lines += [
        "# Isartor — Claude Code + GitHub Copilot Benchmark Report",
        "",
        f"> Generated: {ts}",
        "> Benchmark: Claude Code todo-app fixture (`claude_code_todo.jsonl`)",
        "> Scenarios: baseline (no Isartor) · cold cache · warm cache",
        "",
    ]

    # ── Executive summary ────────────────────────────────────────────────
    lines += [
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Warm-cache deflection rate | **{pct(lb['warm_deflection_rate'])}** |",
        f"| Cold-cache deflection rate | {pct(lb['cold_deflection_rate'])} |",
        f"| P50 latency reduction (baseline → warm) | {latency['p50_reduction_ms']:.1f} ms ({latency['p50_reduction_pct']:.1f}%) |",
        f"| Monthly cloud tokens saved (est.) | {roi_vals['tokens_saved_per_month_est']:,} |",
        f"| Monthly cloud cost saved (est.) | **{usd2(cost['monthly_cloud_saved_usd'])}** |",
        f"| ROI multiple (savings ÷ hosting cost) | **{roi_vals['roi_multiple']:.1f}×** |",
        "",
    ]

    # ── Scenario results ──────────────────────────────────────────────────
    lines += [
        "## Scenario Results",
        "",
        "### Layer Breakdown by Scenario",
        "",
        "| Scenario | Total | L1a (exact) | L1b (semantic) | L2 (SLM) | L3 (cloud) | Errors | Deflection |",
        "|----------|-------|-------------|----------------|----------|------------|--------|------------|",
    ]

    for name in ("baseline", "cold", "warm"):
        s = scenarios.get(name)
        if not s:
            continue
        total = s["total_requests"]
        lines.append(
            f"| {name} "
            f"| {total} "
            f"| {s['l1a_hits']} ({pct(s['l1a_rate'])}) "
            f"| {s['l1b_hits']} ({pct(s['l1b_rate'])}) "
            f"| {s['l2_hits']} ({pct(s['l2_rate'])}) "
            f"| {s['l3_hits']} ({pct(s['l3_rate'])}) "
            f"| {s['error_count']} ({pct(s['error_rate'])}) "
            f"| **{pct(s['deflection_rate'])}** |"
        )

    lines += [
        "",
        "### Latency by Scenario",
        "",
        "| Scenario | P50 | P95 | P99 | L1a P50 | L1b P50 | L2 P50 | L3 P50 |",
        "|----------|-----|-----|-----|---------|---------|--------|--------|",
    ]

    for name in ("baseline", "cold", "warm"):
        s = scenarios.get(name)
        if not s:
            continue
        lines.append(
            f"| {name} "
            f"| {ms(s['p50_ms'])} "
            f"| {ms(s['p95_ms'])} "
            f"| {ms(s['p99_ms'])} "
            f"| {ms(s.get('l1a_p50_ms'))} "
            f"| {ms(s.get('l1b_p50_ms'))} "
            f"| {ms(s.get('l2_p50_ms'))} "
            f"| {ms(s.get('l3_p50_ms'))} |"
        )

    # ── ROI analysis ─────────────────────────────────────────────────────
    lines += [
        "",
        "## ROI Analysis",
        "",
        "### With vs Without Isartor",
        "",
        "| Metric | Without Isartor | With Isartor (warm) | Delta |",
        "|--------|-----------------|---------------------|-------|",
        f"| Cloud cost per request | {usd(cost['cost_per_req_no_isartor_usd'])} | {usd(cost['cost_per_req_with_isartor_usd'])} | -{usd(cost['cost_per_req_no_isartor_usd'] - cost['cost_per_req_with_isartor_usd'])} |",
        f"| Monthly cloud cost ({a['requests_per_month']:,} req) | {usd2(cost['monthly_cloud_cost_without_isartor_usd'])} | {usd2(cost['monthly_cloud_cost_with_isartor_usd'])} | **-{usd2(cost['monthly_cloud_saved_usd'])}** |",
        f"| Monthly Isartor hosting cost | — | {usd2(a['isartor_hosting_cost_per_month_usd'])} | — |",
        f"| Net monthly benefit | — | — | **{usd2(cost['net_monthly_benefit_usd'])}** |",
        f"| ROI multiple | — | — | **{roi_vals['roi_multiple']:.1f}×** |",
        "",
        "### Token Savings",
        "",
        f"- Tokens saved per benchmark run: **{roi_vals['tokens_saved_per_run']:,}**",
        f"- Estimated tokens saved per month ({a['requests_per_month']:,} req): **{roi_vals['tokens_saved_per_month_est']:,}**",
        "",
    ]

    # ── Assumptions ───────────────────────────────────────────────────────
    lines += [
        "## Assumptions",
        "",
        "All cost and ROI figures are estimates based on the following assumptions.",
        "Change these values in `benchmarks/roi_report.py` to match your environment.",
        "",
        "| Assumption | Value | Source |",
        "|------------|-------|--------|",
        f"| gpt-4o input token price | ${a['gpt4o_input_price_per_token_usd']:.6f}/token | OpenAI public pricing (benchmark baseline) |",
        f"| gpt-4o output token price | ${a['gpt4o_output_price_per_token_usd']:.6f}/token | OpenAI public pricing (benchmark baseline) |",
        f"| Avg input tokens per request | {a['avg_input_tokens_per_request']} tokens | Conservative estimate for code prompts |",
        f"| Avg output tokens per response | {a['avg_output_tokens_per_request']} tokens | Typical Claude Code code-generation response |",
        f"| Monthly request volume | {a['requests_per_month']:,} | Workday sessions × typical requests/session |",
        f"| Isartor self-hosting cost/month | ${a['isartor_hosting_cost_per_month_usd']:.0f} | Small VM (2 vCPU, 8 GB RAM) estimate |",
        "| Deflection rate used | warm-cache steady-state | Most representative of production behaviour |",
        "",
        "> **Note:** Only input + output token cloud costs are included.",
        "> Latency benefits, developer productivity gains, and data-privacy",
        "> value (prompts never leave the network for deflected requests)",
        "> are not quantified here but provide additional upside.",
        "",
    ]

    # ── Reproduction instructions ─────────────────────────────────────────
    lines += [
        "## Reproducing This Report",
        "",
        "```bash",
        "# 1. Start Isartor with the Qwen 2.5 Coder 7B sidecar",
        "cd docker",
        "docker compose \\",
        "  -f docker-compose.yml \\",
        "  -f docker-compose.qwen-sidecar.yml \\",
        "  up -d",
        "",
        "# Wait for the Qwen sidecar to finish downloading the model (~4.4 GB)",
        "docker compose -f docker-compose.qwen-sidecar.yml logs -f qwen-sidecar",
        "",
        "# 2. Run the three-scenario benchmark",
        "python3 benchmarks/claude_code_benchmark.py \\",
        "    --url http://localhost:8080 \\",
        "    --api-key changeme",
        "",
        "# 3. Generate this report",
        "python3 benchmarks/roi_report.py",
        "```",
        "",
        "Or with the Makefile shortcuts:",
        "",
        "```bash",
        "make benchmark-claude-code          # run all three scenarios",
        "make benchmark-claude-code-report   # generate ROI report",
        "```",
        "",
    ]

    # ── Footer ────────────────────────────────────────────────────────────
    lines += [
        "---",
        f"*Report generated by `benchmarks/roi_report.py` at {ts}.*",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Isartor Claude Code benchmark ROI report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to benchmark results JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_MD),
        help=f"Path for the Markdown report (default: {DEFAULT_OUTPUT_MD})",
    )
    parser.add_argument(
        "--requests-per-month",
        type=int,
        default=REQUESTS_PER_MONTH_DEFAULT,
        dest="requests_per_month",
        help=f"Monthly request volume for ROI projection (default: {REQUESTS_PER_MONTH_DEFAULT:,})",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        dest="print_only",
        help="Print the report to stdout without writing any files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        print(
            "  Run claude_code_benchmark.py first to generate results,\n"
            "  or use --dry-run to generate a simulated result.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw = json.loads(input_path.read_text())
    scenarios = raw.get("scenarios", {})

    if not scenarios:
        print("[ERROR] No scenario results found in input file.", file=sys.stderr)
        sys.exit(1)

    # ── Compute ROI ───────────────────────────────────────────────────────
    roi = compute_roi(scenarios, requests_per_month=args.requests_per_month)

    # ── Render Markdown ───────────────────────────────────────────────────
    report_md = render_markdown(raw, roi)

    if args.print_only:
        print(report_md)
        return

    # ── Write Markdown report ─────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.write_text(report_md)
    print(f"  Markdown report → {out_path}")

    # ── Write augmented JSON artifact ─────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = RESULTS_DIR / f"claude_code_roi_{ts}.json"
    artifact = {**raw, "roi": roi}
    json_out.write_text(json.dumps(artifact, indent=2))
    print(f"  ROI artifact    → {json_out}")

    # Also update the stable latest ROI path
    latest_roi = RESULTS_DIR / "claude_code_roi_latest.json"
    latest_roi.write_text(json.dumps(artifact, indent=2))
    print(f"  Latest ROI      → {latest_roi}")

    print("\n  Done.")


if __name__ == "__main__":
    main()

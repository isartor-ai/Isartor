#!/usr/bin/env python3
"""
Isartor ROI Report Generator

Produces a side-by-side **with-Isartor vs without-Isartor** analysis from
benchmark data and writes:
  - A machine-readable JSON artifact  (benchmarks/results/roi_report.json)
  - A Markdown report                 (benchmarks/results/roi_report.md)

Usage:
    # Generate from existing benchmark results (benchmarks/results/latest.json):
    python3 benchmarks/report.py

    # Generate from a specific results file:
    python3 benchmarks/report.py --input benchmarks/results/ci_run.json

    # Run a dry-run benchmark first, then generate the report:
    python3 benchmarks/report.py --dry-run

    # Specify output paths:
    python3 benchmarks/report.py \\
        --output-json benchmarks/results/roi_report.json \\
        --output-md   benchmarks/results/roi_report.md

    # Include the Claude Code fixture in the report:
    python3 benchmarks/report.py --include-claude-code

The "without Isartor" baseline is modelled as:
    - 100 % of requests forwarded to L3 (cloud)
    - Latency = measured L3 p50 for every request
    - Token cost = full input + output cost for every request

Assumptions (clearly marked in the report):
    - Average prompt (input) tokens   : 150  (Claude Code typical context)
    - Average output tokens           : 300  (Claude Code typical response)
    - gpt-4o input  price (USD/token) : 0.000005
    - gpt-4o output price (USD/token) : 0.000015
    - L2 SLM: input tokens consumed, output tokens NOT billed externally
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Default paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
DEFAULT_INPUT = RESULTS_DIR / "latest.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "roi_report.json"
DEFAULT_OUTPUT_MD = RESULTS_DIR / "roi_report.md"

# ── Token / cost assumptions ──────────────────────────────────────────────────
# Claude Code prompts include system context; 150 input tokens is conservative.
AVG_INPUT_TOKENS = 150
# Claude Code responses tend to be code blocks; 300 output tokens is typical.
AVG_OUTPUT_TOKENS = 300
# Public gpt-4o pricing (USD per token)
INPUT_PRICE_PER_TOKEN = 0.000005   # $5 per 1M input tokens
OUTPUT_PRICE_PER_TOKEN = 0.000015  # $15 per 1M output tokens


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f} ms"


def _fmt_usd(v: float, precision: int = 4) -> str:
    return f"${v:.{precision}f}"


def _cost_without(total: int) -> dict:
    """Model the full cloud cost if NO Isartor is in the path."""
    input_tokens = total * AVG_INPUT_TOKENS
    output_tokens = total * AVG_OUTPUT_TOKENS
    cost = input_tokens * INPUT_PRICE_PER_TOKEN + output_tokens * OUTPUT_PRICE_PER_TOKEN
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": round(cost, 6),
        "cost_per_req_usd": round(cost / total, 8) if total else 0.0,
    }


def _cost_with(r: dict) -> dict:
    """
    Model the cloud cost WITH Isartor.

    - L1a, L1b: 0 cloud tokens (served from cache)
    - L2:       only input tokens consumed by the SLM (no cloud output billed)
    - L3:       full input + output tokens billed to cloud provider
    """
    total = r["total_requests"]
    l2_hits = r["l2_hits"]
    l3_hits = r["l3_hits"]

    # L3 pays full round-trip
    l3_input = l3_hits * AVG_INPUT_TOKENS
    l3_output = l3_hits * AVG_OUTPUT_TOKENS
    # L2 only uses local SLM; no cloud output tokens are billed
    l2_input = l2_hits * AVG_INPUT_TOKENS

    cloud_input = l3_input + l2_input
    cloud_output = l3_output
    cost = cloud_input * INPUT_PRICE_PER_TOKEN + cloud_output * OUTPUT_PRICE_PER_TOKEN

    return {
        "cloud_input_tokens": cloud_input,
        "cloud_output_tokens": cloud_output,
        "total_cost_usd": round(cost, 6),
        "cost_per_req_usd": round(cost / total, 8) if total else 0.0,
        "l3_input_tokens": l3_input,
        "l3_output_tokens": l3_output,
        "l2_input_tokens": l2_input,
    }


def analyse_fixture(name: str, r: dict) -> dict:
    """Compute the full with/without ROI analysis for one fixture result."""
    total = r["total_requests"]
    if total == 0:
        return {}

    without = _cost_without(total)
    with_ = _cost_with(r)

    tokens_saved_input = without["input_tokens"] - with_["cloud_input_tokens"]
    tokens_saved_output = without["output_tokens"] - with_["cloud_output_tokens"]
    cost_saved = without["total_cost_usd"] - with_["total_cost_usd"]
    cost_saved_pct = cost_saved / without["total_cost_usd"] if without["total_cost_usd"] else 0.0

    # Latency delta: with-Isartor overall p50 vs without (which would be L3 p50
    # for every request, so baseline overall p50 ≈ L3 p50)
    l3_p50 = r.get("l3_p50_ms") or 0.0
    overall_p50_with = r.get("p50_ms", 0.0)
    latency_delta_p50 = l3_p50 - overall_p50_with  # positive = Isartor is faster

    return {
        "fixture": name,
        "total_requests": total,
        # Layer distribution
        "l1a_hits": r["l1a_hits"],
        "l1b_hits": r["l1b_hits"],
        "l2_hits": r["l2_hits"],
        "l3_hits": r["l3_hits"],
        "l1a_rate": r["l1a_rate"],
        "l1b_rate": r["l1b_rate"],
        "l2_rate": r["l2_rate"],
        "l3_rate": r["l3_rate"],
        "deflection_rate": r["deflection_rate"],
        "error_count": r.get("error_count", 0),
        # Latency
        "p50_ms_with": r.get("p50_ms"),
        "p95_ms_with": r.get("p95_ms"),
        "p99_ms_with": r.get("p99_ms"),
        "l1a_p50_ms": r.get("l1a_p50_ms"),
        "l1b_p50_ms": r.get("l1b_p50_ms"),
        "l2_p50_ms": r.get("l2_p50_ms"),
        "l3_p50_ms": r.get("l3_p50_ms"),
        "p50_ms_without": l3_p50,   # baseline: every request goes to L3
        "latency_delta_p50_ms": round(latency_delta_p50, 2),
        # Token analysis
        "without_isartor": without,
        "with_isartor": with_,
        "tokens_saved_input": tokens_saved_input,
        "tokens_saved_output": tokens_saved_output,
        "cost_saved_usd": round(cost_saved, 6),
        "cost_saved_pct": round(cost_saved_pct, 4),
    }


# ── Markdown generation ───────────────────────────────────────────────────────

def _md_fixture_section(analysis: dict) -> str:
    """Render one fixture analysis as a Markdown section."""
    name = analysis["fixture"]
    total = analysis["total_requests"]
    defl_pct = analysis["deflection_rate"]

    # Layer distribution table
    def _row(label, hits, rate, p50_ms):
        return (
            f"| {label:<18} | {hits:>6} | {_fmt_pct(rate):>12} "
            f"| {_fmt_ms(p50_ms):>17} |"
        )

    lines = [
        f"## Fixture: `{name}`",
        "",
        f"> **{total:,} requests** — deflection rate: **{_fmt_pct(defl_pct)}**",
        "",
        "### Layer distribution",
        "",
        "| Layer              |  Hits  | % of Traffic | Avg Latency (p50) |",
        "|--------------------|--------|--------------|-------------------|",
        _row("L1a (exact cache)", analysis["l1a_hits"], analysis["l1a_rate"], analysis["l1a_p50_ms"]),
        _row("L1b (semantic)", analysis["l1b_hits"], analysis["l1b_rate"], analysis["l1b_p50_ms"]),
        _row("L2  (local SLM)", analysis["l2_hits"], analysis["l2_rate"], analysis["l2_p50_ms"]),
        _row("L3  (cloud)", analysis["l3_hits"], analysis["l3_rate"], analysis["l3_p50_ms"]),
        "",
        f"> Overall (with Isartor) — "
        f"P50: {_fmt_ms(analysis['p50_ms_with'])} | "
        f"P95: {_fmt_ms(analysis['p95_ms_with'])} | "
        f"P99: {_fmt_ms(analysis['p99_ms_with'])}",
        "",
        "### With vs without Isartor",
        "",
        "| Metric                          | Without Isartor (baseline) | With Isartor       |",
        "|---------------------------------|----------------------------|--------------------|",
    ]

    wo = analysis["without_isartor"]
    wi = analysis["with_isartor"]
    saved = analysis["cost_saved_usd"]
    saved_pct = analysis["cost_saved_pct"]
    tok_in_saved = analysis["tokens_saved_input"]
    tok_out_saved = analysis["tokens_saved_output"]
    lat_delta = analysis["latency_delta_p50_ms"]
    lat_without = _fmt_ms(analysis["p50_ms_without"])
    lat_with = _fmt_ms(analysis["p50_ms_with"])

    lines += [
        f"| Cloud input tokens              | {wo['input_tokens']:>26,} | {wi['cloud_input_tokens']:>18,} |",
        f"| Cloud output tokens             | {wo['output_tokens']:>26,} | {wi['cloud_output_tokens']:>18,} |",
        f"| Cloud cost (total)              | {_fmt_usd(wo['total_cost_usd'], 4):>26} | {_fmt_usd(wi['total_cost_usd'], 4):>18} |",
        f"| Cloud cost (per request)        | {_fmt_usd(wo['cost_per_req_usd'], 6):>26} | {_fmt_usd(wi['cost_per_req_usd'], 6):>18} |",
        f"| Overall P50 latency             | {lat_without:>26} | {lat_with:>18} |",
        "",
        f"> **Input tokens saved:** {tok_in_saved:,}  "
        f"**Output tokens saved:** {tok_out_saved:,}  "
        f"**Cost saved:** {_fmt_usd(saved, 4)} ({_fmt_pct(saved_pct)} reduction)  "
        f"**Latency delta (P50):** {lat_delta:+.0f} ms (positive = faster with Isartor)",
        "",
    ]
    return "\n".join(lines)


def _md_l2_justification(analyses: list[dict]) -> str:
    """Render the L2 SLM justification section."""
    # Aggregate L2 data across all fixtures
    total_l2 = sum(a["l2_hits"] for a in analyses)
    total_reqs = sum(a["total_requests"] for a in analyses)
    l2_rate = total_l2 / total_reqs if total_reqs else 0.0

    avg_l2_p50_vals = [a["l2_p50_ms"] for a in analyses if a.get("l2_p50_ms") is not None]
    avg_l3_p50_vals = [a["l3_p50_ms"] for a in analyses if a.get("l3_p50_ms") is not None]
    avg_l2_p50 = sum(avg_l2_p50_vals) / len(avg_l2_p50_vals) if avg_l2_p50_vals else None
    avg_l3_p50 = sum(avg_l3_p50_vals) / len(avg_l3_p50_vals) if avg_l3_p50_vals else None

    lines = [
        "## L2 SLM sidecar justification",
        "",
        "The L2 layer runs a **Small Language Model (SLM) sidecar** locally "
        "on the same host as Isartor. It intercepts requests that miss L1a/L1b "
        "and attempts to answer them without reaching the cloud provider.",
        "",
        "### When L2 adds value",
        "",
        "| Scenario                              | L2 verdict |",
        "|---------------------------------------|------------|",
        "| Prompt matches known FAQ / code snippet | ✅ Deflects cloud call at ~100–200 ms |",
        "| Prompt requires deep code generation  | ❌ Falls through to L3 |",
        "| Offline / air-gapped environment      | ✅ Covers requests L1 would miss |",
        "| Low-quality SLM config                | ⚠️  Increases latency without saving cost |",
        "",
        "### Observed contribution",
        "",
    ]

    if total_reqs > 0:
        lines += [
            f"- **{total_l2:,}** L2 deflections across {total_reqs:,} total requests "
            f"({_fmt_pct(l2_rate)} of traffic)",
        ]
    if avg_l2_p50 is not None and avg_l3_p50 is not None:
        speedup = avg_l3_p50 / avg_l2_p50 if avg_l2_p50 > 0 else 0.0
        lines += [
            f"- L2 median latency: **{_fmt_ms(avg_l2_p50)}** vs "
            f"L3 median: **{_fmt_ms(avg_l3_p50)}** "
            f"(L2 is ~{speedup:.1f}× faster when it can answer)",
        ]

    lines += [
        "",
        "### Trade-offs and recommendation",
        "",
        "- Enable L2 (`enable_slm_router = true`) when: high prompt repetition, "
        "offline-capable SLM available, or cost savings are critical.",
        "- Disable L2 (`enable_slm_router = false`) when: all prompts are highly "
        "novel, the SLM quality is insufficient, or added latency for L3-bound "
        "requests outweighs savings.",
        "- Even with L2 disabled, L1a + L1b alone typically achieve **≥ 60%** "
        "deflection on repetitive workloads.",
        "",
    ]
    return "\n".join(lines)


def _md_assumptions() -> str:
    return "\n".join([
        "## Assumptions and methodology",
        "",
        "> ⚠️  **The token and cost figures below are estimates.** "
        "Exact provider token counts are not available from Isartor's "
        "layer-routing headers. The following conservative defaults are used:",
        "",
        f"| Parameter                    | Value    | Rationale                             |",
        f"|------------------------------|----------|---------------------------------------|",
        f"| Average input tokens         | {AVG_INPUT_TOKENS:>8} | Claude Code context + system prompt   |",
        f"| Average output tokens        | {AVG_OUTPUT_TOKENS:>8} | Typical code generation response      |",
        f"| gpt-4o input price (USD/tok) | {INPUT_PRICE_PER_TOKEN} | Public OpenAI pricing                 |",
        f"| gpt-4o output price (USD/tok)| {OUTPUT_PRICE_PER_TOKEN} | Public OpenAI pricing                 |",
        f"| Without-Isartor baseline     | 100% L3  | Every request reaches the cloud       |",
        f"| L2 cost model                | input only | SLM answers locally; no cloud output|",
        "",
        "**Latency baseline (without Isartor):** modelled as the measured L3 p50 "
        "latency applied to every request. Real-world cloud latency varies; "
        "the P50 is a representative central estimate.",
        "",
        "**Error / interruption / rerun delta:** with Isartor, cache-hit requests "
        "(L1a, L1b) are immune to cloud outages and rate-limit errors. "
        "The deflection rate directly represents the fraction of requests "
        "protected from cloud interruptions.",
        "",
    ])


def generate_markdown(
    analyses: list[dict],
    source_file: str,
    hardware: str,
    timestamp: str,
    is_dry_run: bool,
) -> str:
    run_note = (
        "> ⚠️  **Dry-run data** — responses were simulated locally. "
        "Layer distribution mirrors reference numbers from `benchmarks/results/latest.json`. "
        "For production ROI analysis, run against a live Isartor instance.\n\n"
        if is_dry_run
        else ""
    )

    header = "\n".join([
        "# Isartor ROI Report — Claude Code + Copilot",
        "",
        f"> Generated: {timestamp}  ",
        f"> Source data: `{source_file}`  ",
        f"> Hardware: {hardware}  ",
        "",
        run_note,
        "This report compares **Claude Code + GitHub Copilot running through Isartor** "
        "against a **baseline with no Isartor** (all requests forwarded directly to the "
        "cloud provider). It covers L1/L2/L3 layer distribution, token and cost savings, "
        "and latency impact.",
        "",
    ])

    fixture_sections = "\n\n".join(_md_fixture_section(a) for a in analyses)

    # Aggregate summary across all fixtures
    total_requests = sum(a["total_requests"] for a in analyses)
    total_cost_without = sum(a["without_isartor"]["total_cost_usd"] for a in analyses)
    total_cost_with = sum(a["with_isartor"]["total_cost_usd"] for a in analyses)
    total_saved = sum(a["cost_saved_usd"] for a in analyses)
    total_saved_pct = total_saved / total_cost_without if total_cost_without > 0 else 0.0
    total_deflected = sum(
        a["l1a_hits"] + a["l1b_hits"] + a["l2_hits"] for a in analyses
    )
    overall_deflection = total_deflected / total_requests if total_requests else 0.0
    total_tok_in_saved = sum(a["tokens_saved_input"] for a in analyses)
    total_tok_out_saved = sum(a["tokens_saved_output"] for a in analyses)

    summary = "\n".join([
        "## Executive summary",
        "",
        f"| Metric                     | Value                              |",
        f"|----------------------------|------------------------------------|",
        f"| Total requests analysed    | {total_requests:>34,} |",
        f"| Overall deflection rate    | {_fmt_pct(overall_deflection):>34} |",
        f"| Cloud input tokens saved   | {total_tok_in_saved:>34,} |",
        f"| Cloud output tokens saved  | {total_tok_out_saved:>34,} |",
        f"| Estimated cost (without)   | {_fmt_usd(total_cost_without, 4):>34} |",
        f"| Estimated cost (with)      | {_fmt_usd(total_cost_with, 4):>34} |",
        f"| Estimated cost saved       | {_fmt_usd(total_saved, 4)} ({_fmt_pct(total_saved_pct)} reduction) |",
        "",
    ])

    l2_section = _md_l2_justification(analyses)
    assumptions = _md_assumptions()

    footer = "\n".join([
        "---",
        "",
        "_Report generated by `benchmarks/report.py`. "
        "Re-run with `make report` (live server) or `make report-dry-run` (offline)._",
        "",
    ])

    return "\n\n".join([header, summary, fixture_sections, l2_section, assumptions, footer])


# ── Benchmark runner (dry-run) ────────────────────────────────────────────────

def _run_dry_run_benchmark() -> dict:
    """Run the benchmark harness in dry-run mode and return the results dict."""
    import subprocess
    output_path = RESULTS_DIR / "roi_dryrun.json"
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run.py"),
        "--all",
        "--dry-run",
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Benchmark dry-run failed")
    print(result.stdout)
    with output_path.open() as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isartor ROI Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Path to an existing benchmark results JSON file "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Run a fresh dry-run benchmark before generating the report. "
            "Useful in CI where no live server is available."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        dest="output_json",
        help=f"Path for the machine-readable JSON artifact (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        dest="output_md",
        help=f"Path for the Markdown report (default: {DEFAULT_OUTPUT_MD})",
    )
    args = parser.parse_args()

    # ── Load benchmark data ────────────────────────────────────────────────
    is_dry_run = args.dry_run
    if is_dry_run:
        print("Running dry-run benchmark …")
        bench_data = _run_dry_run_benchmark()
        source_label = "dry-run simulation"
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(
                f"Error: benchmark results file not found: {input_path}\n"
                "Run `make benchmark` (live server) or use --dry-run.",
                file=sys.stderr,
            )
            sys.exit(1)
        with input_path.open() as f:
            bench_data = json.load(f)
        source_label = str(input_path)

    fixtures: dict[str, dict] = bench_data.get("fixtures", {})
    if not fixtures:
        print("Error: no fixture results found in benchmark data.", file=sys.stderr)
        sys.exit(1)

    hardware = bench_data.get("hardware", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Analyse each fixture ───────────────────────────────────────────────
    analyses = []
    for name, result in fixtures.items():
        a = analyse_fixture(name, result)
        if a:
            analyses.append(a)
            _print_analysis(a)

    if not analyses:
        print("Error: no valid fixture analyses produced.", file=sys.stderr)
        sys.exit(1)

    # ── Write JSON artifact ────────────────────────────────────────────────
    json_out = Path(args.output_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "1",
        "report_type": "roi_comparison",
        "generated_at": timestamp,
        "source_data": source_label,
        "hardware": hardware,
        "is_dry_run": is_dry_run,
        "assumptions": {
            "avg_input_tokens": AVG_INPUT_TOKENS,
            "avg_output_tokens": AVG_OUTPUT_TOKENS,
            "input_price_per_token_usd": INPUT_PRICE_PER_TOKEN,
            "output_price_per_token_usd": OUTPUT_PRICE_PER_TOKEN,
            "baseline_model": "all requests reach L3 (cloud)",
        },
        "fixtures": {a["fixture"]: a for a in analyses},
        "aggregate": {
            "total_requests": sum(a["total_requests"] for a in analyses),
            "overall_deflection_rate": round(
                sum(a["l1a_hits"] + a["l1b_hits"] + a["l2_hits"] for a in analyses)
                / max(sum(a["total_requests"] for a in analyses), 1),
                4,
            ),
            "total_tokens_saved_input": sum(a["tokens_saved_input"] for a in analyses),
            "total_tokens_saved_output": sum(a["tokens_saved_output"] for a in analyses),
            "total_cost_without_usd": round(
                sum(a["without_isartor"]["total_cost_usd"] for a in analyses), 6
            ),
            "total_cost_with_usd": round(
                sum(a["with_isartor"]["total_cost_usd"] for a in analyses), 6
            ),
            "total_cost_saved_usd": round(sum(a["cost_saved_usd"] for a in analyses), 6),
        },
    }
    json_out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nJSON artifact written to {json_out}")

    # ── Write Markdown report ──────────────────────────────────────────────
    md_out = Path(args.output_md)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md = generate_markdown(analyses, source_label, hardware, timestamp, is_dry_run)
    md_out.write_text(md)
    print(f"Markdown report  written to {md_out}")


def _print_analysis(a: dict) -> None:
    """Print a human-readable summary of a fixture analysis to stdout."""
    name = a["fixture"]
    total = a["total_requests"]
    print(f"\n{'─' * 60}")
    print(f"  {name}  ({total:,} requests)")
    print(f"{'─' * 60}")
    print(f"  Deflection rate : {_fmt_pct(a['deflection_rate'])}")
    print(f"  L1a (exact)     : {a['l1a_hits']:>6,}  ({_fmt_pct(a['l1a_rate'])})")
    print(f"  L1b (semantic)  : {a['l1b_hits']:>6,}  ({_fmt_pct(a['l1b_rate'])})")
    print(f"  L2  (SLM)       : {a['l2_hits']:>6,}  ({_fmt_pct(a['l2_rate'])})")
    print(f"  L3  (cloud)     : {a['l3_hits']:>6,}  ({_fmt_pct(a['l3_rate'])})")
    print()
    wo = a["without_isartor"]
    wi = a["with_isartor"]
    print(f"  Cloud cost (without Isartor) : {_fmt_usd(wo['total_cost_usd'], 4)}")
    print(f"  Cloud cost (with Isartor)    : {_fmt_usd(wi['total_cost_usd'], 4)}")
    print(f"  Cost saved                   : {_fmt_usd(a['cost_saved_usd'], 4)} "
          f"({_fmt_pct(a['cost_saved_pct'])} reduction)")
    print(f"  Latency delta (P50)          : {a['latency_delta_p50_ms']:+.0f} ms "
          "(positive = Isartor faster)")


if __name__ == "__main__":
    main()

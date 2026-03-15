#!/usr/bin/env python3
"""
Isartor Benchmark Harness

Usage:
    # Run both built-in fixtures and write results:
    python benchmarks/run.py --url http://localhost:8080 --all

    # Run a single fixture with an optional request cap:
    python benchmarks/run.py \\
        --url http://localhost:8080 \\
        --input benchmarks/fixtures/faq_loop.jsonl \\
        --requests 1000

    # Dry-run (no server required — uses simulated responses for CI):
    python benchmarks/run.py --all --dry-run

    # Honour environment variables:
    ISARTOR_URL=http://localhost:3000 \\
    ISARTOR_API_KEY=changeme \\
    python benchmarks/run.py --all
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
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"

# gpt-4o input token price (USD per token)
GPT4O_INPUT_PRICE_PER_TOKEN = 0.000005
# Average prompt token estimate used for cost calculation
AVG_PROMPT_TOKENS = 50

LAYERS = ("l1a", "l1b", "l2", "l3")


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
    """Return the *pct*-th percentile of an already-sorted list."""
    if not sorted_data:
        return 0.0
    # Use ceil-based index so that e.g. P99 on 1000 items maps to index 989
    # (the 990th element), matching the "nearest-rank" definition.
    idx = min(math.ceil(len(sorted_data) * pct / 100) - 1, len(sorted_data) - 1)
    return sorted_data[max(idx, 0)]


def _layer_p50(layer_latencies: dict[str, list[float]], layer: str) -> str:
    """Return formatted p50 latency for a layer, or '-' if no data."""
    lats = layer_latencies.get(layer, [])
    if not lats:
        return "-"
    return f"{statistics.median(lats):.1f} ms"


def _simulate_response(prompt: str) -> tuple[str, float]:
    """
    Simulate an Isartor response for dry-run / CI mode.

    Returns (layer, latency_ms).  Distribution roughly mirrors the reference
    numbers in results/latest.json so that CI output is meaningful.

    Uses a per-prompt seeded RNG derived from a stable MD5 hash so that
    results are fully deterministic: the same JSONL corpus always produces
    the same output regardless of Python's PYTHONHASHSEED or run order.
    """
    # Stable 16-bit fingerprint from MD5 (not security-sensitive)
    digest = hashlib.md5(prompt.encode(), usedforsecurity=False).digest()
    h = int.from_bytes(digest[:2], "little")
    rng = random.Random(int.from_bytes(digest, "little"))
    if h < 0x6800:        # ~41 % -> L1a
        layer, latency = "l1a", rng.uniform(0.1, 0.6)
    elif h < 0xAB00:      # ~27 % -> L1b
        layer, latency = "l1b", rng.uniform(1.0, 5.0)
    elif h < 0xBC00:      # ~7 % -> L2
        layer, latency = "l2", rng.uniform(50.0, 200.0)
    else:                 # ~25 % -> L3
        layer, latency = "l3", rng.uniform(400.0, 1200.0)
    return layer, latency


def run_benchmark(
    url: str,
    prompts: list[str],
    label: str,
    *,
    dry_run: bool = False,
    api_key: str = "changeme",
    timeout: float = 120.0,
) -> dict:
    """
    Send each prompt to ``url/api/chat`` and collect per-request statistics.

    The server is expected to return the ``X-Isartor-Layer`` response header
    with one of the values: ``l1a``, ``l1b``, ``l2``, ``l3``.

    Parameters
    ----------
    url:      Base URL of the Isartor instance.
    prompts:  List of prompt strings to replay.
    label:    Human-readable name used in printed output.
    dry_run:  When *True*, requests are simulated locally — no server needed.
    api_key:  Value for the ``X-API-Key`` header.  Must match the server's
              ``gateway_api_key`` setting (default: ``"changeme"``, which is
              the server's own config default from ``src/config.rs``).
    timeout:  Per-request timeout in seconds (default: 120).  Azure OpenAI
              L3 calls can take 20–60 s on cold starts, so 120 s gives ample
              headroom while still catching genuine hangs.
    """
    results: dict[str, int] = {"l1a": 0, "l1b": 0, "l2": 0, "l3": 0, "error": 0}
    # Per-layer latency lists for per-layer p50 in the table
    layer_latencies: dict[str, list[float]] = {k: [] for k in LAYERS}
    # All latencies for overall p50/p95/p99
    all_latencies: list[float] = []

    for prompt in prompts:
        if dry_run:
            layer, latency_ms = _simulate_response(prompt)
            # Normalise/validate simulated layer values.
            if layer not in LAYERS and layer != "error":
                results["error"] += 1
                print(
                    f"  [warn] unexpected simulated layer value: {layer!r}; "
                    "counting as error and excluding from latency stats.",
                    file=sys.stderr,
                )
                continue
            if layer == "error":
                results["error"] += 1
                # Do not include error responses in latency statistics.
                continue
            results[layer] = results.get(layer, 0) + 1
            layer_latencies[layer].append(latency_ms)
            all_latencies.append(latency_ms)
            continue

        start = time.perf_counter()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        req = urllib.request.Request(
            f"{url}/api/chat",
            data=json.dumps({"prompt": prompt}).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_layer = resp.headers.get("X-Isartor-Layer", "l3")
                latency_ms = (time.perf_counter() - start) * 1000
                # Normalise/validate header value. Map unknown values to "error"
                # and exclude them from latency statistics to keep summaries
                # consistent with the rendered table and deflection metrics.
                if raw_layer not in LAYERS and raw_layer != "error":
                    results["error"] += 1
                    print(
                        "  [warn] unexpected X-Isartor-Layer header value "
                        f"{raw_layer!r}; counting as error and excluding "
                        "from latency stats.",
                        file=sys.stderr,
                    )
                    continue
                if raw_layer == "error":
                    results["error"] += 1
                    # Do not include error responses in latency statistics.
                    continue
                layer = raw_layer
                results[layer] = results.get(layer, 0) + 1
                layer_latencies[layer].append(latency_ms)
                all_latencies.append(latency_ms)
        except urllib.error.HTTPError as exc:
            results["error"] += 1
            if exc.code == 401:
                print(
                    "  [warn] 401 Unauthorized — set --api-key / $ISARTOR_API_KEY "
                    "to match the server's gateway_api_key (default: 'changeme')",
                    file=sys.stderr,
                )
            else:
                print(f"  [warn] HTTP {exc.code}: {exc}", file=sys.stderr)
        except urllib.error.URLError as exc:
            results["error"] += 1
            print(f"  [warn] request failed: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            results["error"] += 1
            print(f"  [warn] unexpected error: {exc}", file=sys.stderr)

    total = len(prompts)

    if total == 0:
        print(f"\n-- {label} --")
        print("  No prompts to run; skipping benchmark.")
        return _empty_result()

    deflected = results["l1a"] + results["l1b"] + results["l2"]
    deflection_pct = deflected / total * 100

    # ── Latency percentiles (overall) ──────────────────────────────────────
    sorted_all = sorted(all_latencies)
    p50 = statistics.median(all_latencies) if all_latencies else 0.0
    p95 = _percentile(sorted_all, 95) if all_latencies else 0.0
    p99 = _percentile(sorted_all, 99) if all_latencies else 0.0

    # ── Cost savings ───────────────────────────────────────────────────────
    # tokens_saved = avg_prompt_tokens * (l1a_hits + l1b_hits + l2_hits)
    # cost_saved   = tokens_saved * gpt4o_input_price_per_token
    tokens_saved = AVG_PROMPT_TOKENS * deflected
    total_cost_saved = tokens_saved * GPT4O_INPUT_PRICE_PER_TOKEN
    cost_per_req = (total_cost_saved / total) if total else 0.0

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n-- {label} --")
    print(f"  Total requests : {total}")
    print(
        f"  L1a (exact)    : {results['l1a']:5d}  ({results['l1a'] / total * 100:.1f}%)"
    )
    print(
        f"  L1b (semantic) : {results['l1b']:5d}  ({results['l1b'] / total * 100:.1f}%)"
    )
    print(
        f"  L2  (SLM)      : {results['l2']:5d}  ({results['l2'] / total * 100:.1f}%)"
    )
    print(
        f"  L3  (cloud)    : {results['l3']:5d}  ({results['l3'] / total * 100:.1f}%)"
    )
    print(f"  Errors         : {results['error']:5d}")
    print(f"  Deflection rate: {deflection_pct:.1f}%")
    print(f"  P50 latency    : {p50:.1f} ms")
    print(f"  P95 latency    : {p95:.1f} ms")
    print(f"  P99 latency    : {p99:.1f} ms")
    print(f"  Cost saved     : ${total_cost_saved:.4f}  (${cost_per_req:.6f}/req)")

    # ── Markdown table (copy-pasteable) ────────────────────────────────────
    print()
    print(_markdown_table(
        label, total, results, layer_latencies,
        p50, p95, p99, deflection_pct, cost_per_req,
    ))

    # ── Per-layer p50 latencies (for JSON output and PR comment table) ────────
    def layer_p50_val(layer: str) -> float | None:
        lats = layer_latencies.get(layer, [])
        return round(statistics.median(lats), 2) if lats else None

    return {
        "total_requests": total,
        "deflection_rate": round(deflected / total, 4),
        "l1a_hits": results["l1a"],
        "l1b_hits": results["l1b"],
        "l2_hits": results["l2"],
        "l3_hits": results["l3"],
        "l1a_rate": round(results["l1a"] / total, 4),
        "l1b_rate": round(results["l1b"] / total, 4),
        "l2_rate": round(results["l2"] / total, 4),
        "l3_rate": round(results["l3"] / total, 4),
        "error_count": results["error"],
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "l1a_p50_ms": layer_p50_val("l1a"),
        "l1b_p50_ms": layer_p50_val("l1b"),
        "l2_p50_ms": layer_p50_val("l2"),
        "l3_p50_ms": layer_p50_val("l3"),
        "tokens_saved": tokens_saved,
        "cost_saved_usd": round(total_cost_saved, 6),
        "cost_per_req_usd": round(cost_per_req, 8),
    }


def _empty_result() -> dict:
    return {
        "total_requests": 0,
        "deflection_rate": 0.0,
        "l1a_hits": 0,
        "l1b_hits": 0,
        "l2_hits": 0,
        "l3_hits": 0,
        "l1a_rate": 0.0,
        "l1b_rate": 0.0,
        "l2_rate": 0.0,
        "l3_rate": 0.0,
        "error_count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
        "l1a_p50_ms": None,
        "l1b_p50_ms": None,
        "l2_p50_ms": None,
        "l3_p50_ms": None,
        "tokens_saved": 0,
        "cost_saved_usd": 0.0,
        "cost_per_req_usd": 0.0,
    }


def _markdown_table(
    label: str,
    total: int,
    results: dict,
    layer_latencies: dict[str, list[float]],
    p50: float,
    p95: float,
    p99: float,
    deflection_pct: float,
    cost_per_req: float,
) -> str:
    """Return a copy-pasteable Markdown result table."""
    l1a_pct = results["l1a"] / total * 100
    l1b_pct = results["l1b"] / total * 100
    l2_pct = results["l2"] / total * 100
    l3_pct = results["l3"] / total * 100
    deflected = results["l1a"] + results["l1b"] + results["l2"]

    l1a_lat = _layer_p50(layer_latencies, "l1a")
    l1b_lat = _layer_p50(layer_latencies, "l1b")
    l2_lat  = _layer_p50(layer_latencies, "l2")
    l3_lat  = _layer_p50(layer_latencies, "l3")

    lines = [
        f"### {label}",
        "",
        "| Layer              | Hits   | % of Traffic | Avg Latency (p50) |",
        "|--------------------|--------|--------------|-------------------|",
        f"| L1a (exact)        | {results['l1a']:6d} | {l1a_pct:>10.1f}%  | {l1a_lat:>17} |",
        f"| L1b (semantic)     | {results['l1b']:6d} | {l1b_pct:>10.1f}%  | {l1b_lat:>17} |",
        f"| L2  (SLM)          | {results['l2']:6d} | {l2_pct:>10.1f}%  | {l2_lat:>17} |",
        f"| L3  (cloud)        | {results['l3']:6d} | {l3_pct:>10.1f}%  | {l3_lat:>17} |",
        f"| **Total deflected**| **{deflected}** | **{deflection_pct:.1f}%** | |",
        f"| **Cost saved**     |        |              | **${cost_per_req:.6f}/req** |",
        "",
        f"> Overall latency — P50: {p50:.1f} ms | P95: {p95:.1f} ms | P99: {p99:.1f} ms",
        f">",
        f"> Methodology: {total} requests replayed sequentially. "
        f"Cost formula: `tokens_saved = {AVG_PROMPT_TOKENS} × (L1a + L1b + L2 hits); "
        f"cost_saved = tokens_saved × {GPT4O_INPUT_PRICE_PER_TOKEN}` (gpt-4o input rate).",
    ]
    return "\n".join(lines)


def hardware_summary() -> str:
    """Best-effort hardware description for the results file."""
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
            f"{mem_gb} RAM, no GPU"
        )
    except Exception:  # noqa: BLE001
        return "unknown hardware"


def write_results(fixture_results: dict[str, dict], output_path: Path) -> None:
    """Write aggregated benchmark results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "isartor_version": "0.1.0",
        "hardware": hardware_summary(),
        "fixtures": fixture_results,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isartor Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_url = os.environ.get("ISARTOR_URL", "http://localhost:8080")
    # The server's gateway_api_key defaults to "changeme" (src/config.rs).
    # $ISARTOR_API_KEY lets CI and local runs override without touching argv.
    default_api_key = os.environ.get("ISARTOR_API_KEY", "changeme")
    parser.add_argument(
        "--url",
        default=default_url,
        help=(
            "Base URL of the running Isartor instance "
            "(default: $ISARTOR_URL or http://localhost:8080)"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        dest="api_key",
        help=(
            "Value for the X-API-Key header sent with every request. "
            "Must match the server's gateway_api_key setting "
            "(default: $ISARTOR_API_KEY or 'changeme')"
        ),
    )
    parser.add_argument(
        "--input",
        help="Path to a JSONL fixture file to benchmark",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="Limit the number of prompts to send (0 = all prompts in file)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all built-in fixtures (faq_loop + diverse_tasks) and write results",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "latest.json"),
        help="Path for the results JSON file (default: benchmarks/results/latest.json)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ISARTOR_TIMEOUT", "120")),
        help=(
            "Per-request timeout in seconds "
            "(default: $ISARTOR_TIMEOUT or 120)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Simulate responses locally without a running server. "
            "Useful for CI validation and smoke-testing."
        ),
    )
    args = parser.parse_args()

    if not args.input and not args.run_all:
        parser.print_help()
        print(
            "\nError: specify --input <file.jsonl> or use --all to run all fixtures.",
            file=sys.stderr,
        )
        sys.exit(1)

    fixture_results: dict[str, dict] = {}

    if args.run_all:
        fixtures = [
            ("faq_loop", FIXTURES_DIR / "faq_loop.jsonl"),
            ("diverse_tasks", FIXTURES_DIR / "diverse_tasks.jsonl"),
        ]
        for name, path in fixtures:
            if not path.exists():
                print(f"[warn] fixture not found, skipping: {path}", file=sys.stderr)
                continue
            prompts = load_prompts(path)
            print(f"\nLoaded {len(prompts)} prompts from {path.name}")
            fixture_results[name] = run_benchmark(
                args.url, prompts, name, dry_run=args.dry_run, api_key=args.api_key,
                timeout=args.timeout,
            )
        write_results(fixture_results, Path(args.output))

    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input fixture file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        prompts = load_prompts(input_path)
        if args.requests > 0:
            prompts = prompts[: args.requests]
        label = input_path.stem
        fixture_results[label] = run_benchmark(
            args.url, prompts, label, dry_run=args.dry_run, api_key=args.api_key,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()

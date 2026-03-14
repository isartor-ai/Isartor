#!/usr/bin/env python3
"""
Isartor Benchmark Harness

Usage:
    python benchmarks/run.py --url http://localhost:3000 \\
                              --input benchmarks/fixtures/faq_loop.jsonl \\
                              --requests 1000

    # Run both built-in fixtures and write results:
    python benchmarks/run.py --url http://localhost:8080 --all
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"


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


def run_benchmark(url: str, prompts: list[str], label: str) -> dict:
    """
    Send each prompt to ``url/api/v1/chat`` and collect per-request statistics.

    The server is expected to return the ``X-Isartor-Layer`` response header
    with one of the values: ``l1a``, ``l1b``, ``l2``, ``l3``.
    """
    results: dict[str, int] = {"l1a": 0, "l1b": 0, "l2": 0, "l3": 0, "error": 0}
    latencies: list[float] = []

    for prompt in prompts:
        start = time.perf_counter()
        req = urllib.request.Request(
            f"{url}/api/v1/chat",
            data=json.dumps({"prompt": prompt}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                layer = resp.headers.get("X-Isartor-Layer", "l3")
                results[layer] = results.get(layer, 0) + 1
                latencies.append((time.perf_counter() - start) * 1000)
        except urllib.error.URLError as exc:
            results["error"] += 1
            print(f"  [warn] request failed: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            results["error"] += 1
            print(f"  [warn] unexpected error: {exc}", file=sys.stderr)

    total = len(prompts)
    deflected = results["l1a"] + results["l1b"] + results["l2"]

    print(f"\n── {label} ──")
    print(f"  Total requests : {total}")
    print(
        f"  L1a (exact)    : {results['l1a']:4d}  "
        f"({results['l1a'] / total * 100:.1f}%)"
    )
    print(
        f"  L1b (semantic) : {results['l1b']:4d}  "
        f"({results['l1b'] / total * 100:.1f}%)"
    )
    print(
        f"  L2  (SLM)      : {results['l2']:4d}  "
        f"({results['l2'] / total * 100:.1f}%)"
    )
    print(
        f"  L3  (cloud)    : {results['l3']:4d}  "
        f"({results['l3'] / total * 100:.1f}%)"
    )
    print(f"  Errors         : {results['error']:4d}")
    print(f"  Deflection rate: {deflected / total * 100:.1f}%")

    if latencies:
        sorted_lat = sorted(latencies)
        p50 = statistics.median(latencies)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        print(f"  P50 latency    : {p50:.1f} ms")
        print(f"  P95 latency    : {p95:.1f} ms")
    else:
        p50 = 0.0
        p95 = 0.0

    return {
        "total_requests": total,
        "deflection_rate": round(deflected / total, 4),
        "l1a_rate": round(results["l1a"] / total, 4),
        "l1b_rate": round(results["l1b"] / total, 4),
        "l2_rate": round(results["l2"] / total, 4),
        "l3_rate": round(results["l3"] / total, 4),
        "error_count": results["error"],
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
    }


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
        return f"{cpu_count}-core {platform.processor() or platform.machine()}, {mem_gb} RAM, no GPU"
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
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the running Isartor instance (default: http://localhost:8080)",
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
            fixture_results[name] = run_benchmark(args.url, prompts, name)
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
        fixture_results[label] = run_benchmark(args.url, prompts, label)


if __name__ == "__main__":
    main()

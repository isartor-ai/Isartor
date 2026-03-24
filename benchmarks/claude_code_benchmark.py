#!/usr/bin/env python3
"""Claude Code three-way benchmark harness.

This harness measures three scenarios for a deterministic TypeScript todo-app
fixture:

- Baseline: direct cloud LLM requests without Isartor.
- Cold cache: first pass through Isartor.
- Warm cache: second pass through Isartor.

It supports fully deterministic dry-run mode for CI and a live mode that sends
Anthropic-compatible requests to either a direct provider or a running Isartor
instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "claude_code_todo_app.jsonl"

CLAUDE_INPUT_PRICE_PER_TOKEN = 0.000003
CLAUDE_OUTPUT_PRICE_PER_TOKEN = 0.000015
AVG_INPUT_TOKENS = 800
AVG_OUTPUT_TOKENS = 200
RETRYABLE_HTTP_STATUSES = {429, 502, 503, 504}
MAX_HTTP_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_SECS = 1.5
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_COMPLETIONS_URL = "https://api.githubcopilot.com/chat/completions"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.29.1"
COPILOT_EDITOR_VERSION = "vscode/1.99.0"

SCENARIO_TO_KEY = {
    "baseline": "baseline",
    "cold": "isartor_cold",
    "warm": "isartor_warm",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hardware_summary() -> str:
    cpu = os.cpu_count() or 1
    machine = platform.machine() or "unknown"
    return f"{cpu}-core {machine}"


def load_fixture(path: Path, limit: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        entries.append(item)
        if limit > 0 and len(entries) >= limit:
            break
    return entries


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct / 100.0))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def stable_rng(seed: str) -> random.Random:
    digest = hashlib.md5(seed.encode("utf-8"), usedforsecurity=False).digest()
    return random.Random(int.from_bytes(digest, "little"))


def simulate_layer(prompt: str, scenario: str) -> tuple[str, float]:
    rng = stable_rng(f"{scenario}:{prompt}")
    draw = rng.random()
    if scenario == "baseline":
        if draw < 0.80:
            return "l3", rng.uniform(800.0, 1800.0)
        if draw < 0.95:
            return "l3", rng.uniform(1800.0, 3500.0)
        return "l3", rng.uniform(3500.0, 8000.0)

    if scenario == "cold":
        if draw < 0.12:
            return "l1a", rng.uniform(0.1, 0.8)
        if draw < 0.20:
            return "l1b", rng.uniform(1.0, 8.0)
        if draw < 0.35:
            return "l2", rng.uniform(50.0, 350.0)
        return "l3", rng.uniform(800.0, 2500.0)

    if draw < 0.45:
        return "l1a", rng.uniform(0.1, 0.8)
    if draw < 0.65:
        return "l1b", rng.uniform(1.0, 8.0)
    if draw < 0.75:
        return "l2", rng.uniform(50.0, 350.0)
    return "l3", rng.uniform(800.0, 2500.0)


def anthropic_request(
    url: str,
    api_key: str,
    prompt: str,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str]]:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": AVG_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                headers_out = {k.lower(): v for k, v in resp.headers.items()}
                resp.read()
                return status, headers_out
        except urllib.error.HTTPError as exc:
            if attempt >= MAX_HTTP_ATTEMPTS or exc.code not in RETRYABLE_HTTP_STATUSES:
                raise
            retry_after = exc.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_secs = max(float(retry_after), HTTP_RETRY_BACKOFF_SECS * attempt)
            else:
                sleep_secs = HTTP_RETRY_BACKOFF_SECS * attempt
            print(
                f"[retry] HTTP {exc.code} from {url} on attempt {attempt}/{MAX_HTTP_ATTEMPTS}; "
                f"retrying in {sleep_secs:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_secs)


def exchange_copilot_session_token(github_token: str, timeout: float) -> str:
    req = urllib.request.Request(
        COPILOT_TOKEN_URL,
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/json",
            "User-Agent": COPILOT_USER_AGENT,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError("direct Copilot token rejected: no active GitHub Copilot subscription found for the supplied token") from exc
        if exc.code == 401:
            raise RuntimeError("direct Copilot token rejected: invalid GitHub token") from exc
        raise RuntimeError(f"direct Copilot session token request returned HTTP {exc.code}") from exc
    token = payload.get("token")
    if not token:
        raise RuntimeError(f"Copilot session token response missing token: {payload}")
    return str(token)


def is_direct_copilot_url(url: str) -> bool:
    return "api.githubcopilot.com" in urlparse(url).netloc


def copilot_request(prompt: str, github_token: str, timeout: float) -> tuple[int, dict[str, str]]:
    session_token = exchange_copilot_session_token(github_token, timeout)
    body = json.dumps(
        {
            "model": "gpt-4o-mini",
            "max_tokens": AVG_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        COPILOT_COMPLETIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {session_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": COPILOT_USER_AGENT,
            "Editor-Version": COPILOT_EDITOR_VERSION,
            "Editor-Plugin-Version": "copilot-chat/0.29.1",
            "Copilot-Integration-Id": "vscode-chat",
            "X-GitHub-Api-Version": "2025-04-01",
        },
        method="POST",
    )

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                headers_out = {k.lower(): v for k, v in resp.headers.items()}
                resp.read()
                return status, headers_out
        except urllib.error.HTTPError as exc:
            if attempt >= MAX_HTTP_ATTEMPTS or exc.code not in RETRYABLE_HTTP_STATUSES:
                raise
            sleep_secs = HTTP_RETRY_BACKOFF_SECS * attempt
            print(
                f"[retry] HTTP {exc.code} from direct Copilot on attempt {attempt}/{MAX_HTTP_ATTEMPTS}; "
                f"retrying in {sleep_secs:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_secs)


def run_scenario(name: str, entries: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not args.dry_run:
        if name == "baseline" and not args.direct_api_key:
            raise SystemExit("baseline live mode requires --direct-api-key or a direct provider token")
        if name in {"cold", "warm"} and not args.api_key:
            raise SystemExit("Isartor live mode requires --api-key or ISARTOR_API_KEY")

    layer_latencies: dict[str, list[float]] = {"l1a": [], "l1b": [], "l2": [], "l3": []}
    errors = 0

    for entry in entries:
        prompt = entry["prompt"]
        start = time.perf_counter()
        try:
            if args.dry_run:
                layer, latency_ms = simulate_layer(prompt, name)
            elif name == "baseline":
                if is_direct_copilot_url(args.direct_url):
                    copilot_request(prompt, args.direct_api_key, args.timeout)
                else:
                    url = args.direct_url.rstrip("/") + "/v1/messages"
                    anthropic_request(url, args.direct_api_key, prompt, args.timeout)
                layer = "l3"
                latency_ms = (time.perf_counter() - start) * 1000.0
            else:
                url = args.isartor_url.rstrip("/") + "/v1/messages"
                _, headers = anthropic_request(
                    url,
                    args.api_key,
                    prompt,
                    args.timeout,
                    extra_headers={"X-API-Key": args.api_key},
                )
                raw_layer = headers.get("x-isartor-layer", "l3").lower().replace("-", "")
                layer = raw_layer if raw_layer in layer_latencies else "l3"
                latency_ms = (time.perf_counter() - start) * 1000.0
            layer_latencies[layer].append(latency_ms)
        except urllib.error.HTTPError as exc:
            if name == "baseline":
                raise SystemExit(f"baseline direct provider returned HTTP {exc.code}: {exc}") from exc
            errors += 1
            print(f"[warn] {name} HTTP {exc.code}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            if name == "baseline":
                raise SystemExit(f"baseline direct provider request failed: {exc}") from exc
            errors += 1
            print(f"[warn] {name} request failed: {exc}", file=sys.stderr)

    total_requests = len(entries)
    all_latencies = [v for values in layer_latencies.values() for v in values]
    l1a_hits = len(layer_latencies["l1a"])
    l1b_hits = len(layer_latencies["l1b"])
    l2_hits = len(layer_latencies["l2"])
    l3_hits = len(layer_latencies["l3"])
    deflected = l1a_hits + l1b_hits + l2_hits
    deflection_rate = (deflected / total_requests) if total_requests else 0.0
    cloud_input_tokens = l3_hits * AVG_INPUT_TOKENS
    cloud_output_tokens = l3_hits * AVG_OUTPUT_TOKENS
    total_cost_usd = cloud_input_tokens * CLAUDE_INPUT_PRICE_PER_TOKEN + cloud_output_tokens * CLAUDE_OUTPUT_PRICE_PER_TOKEN
    cost_per_req_usd = (total_cost_usd / total_requests) if total_requests else 0.0

    return {
        "scenario": name,
        "label": "without_isartor" if name == "baseline" else f"with_isartor_{name}",
        "total_requests": total_requests,
        "deflection_rate": round(deflection_rate, 6),
        "l1a_hits": l1a_hits,
        "l1b_hits": l1b_hits,
        "l2_hits": l2_hits,
        "l3_hits": l3_hits,
        "error_count": errors,
        "p50_ms": round(statistics.median(all_latencies), 3) if all_latencies else 0.0,
        "p95_ms": round(percentile(all_latencies, 95), 3) if all_latencies else 0.0,
        "p99_ms": round(percentile(all_latencies, 99), 3) if all_latencies else 0.0,
        "l1a_p50_ms": round(statistics.median(layer_latencies["l1a"]), 3) if layer_latencies["l1a"] else None,
        "l1b_p50_ms": round(statistics.median(layer_latencies["l1b"]), 3) if layer_latencies["l1b"] else None,
        "l2_p50_ms": round(statistics.median(layer_latencies["l2"]), 3) if layer_latencies["l2"] else None,
        "l3_p50_ms": round(statistics.median(layer_latencies["l3"]), 3) if layer_latencies["l3"] else None,
        "total_cost_usd": round(total_cost_usd, 6),
        "cost_per_req_usd": round(cost_per_req_usd, 8),
    }


def render_report(data: dict[str, Any]) -> str:
    baseline = data.get("baseline")
    cold = data.get("isartor_cold")
    warm = data.get("isartor_warm")
    lines = [
        "# Claude Code + GitHub Copilot - Isartor Benchmark Report",
        "",
        f"**Date:** {data['timestamp']}  ",
        f"**Fixture:** `{data['fixture']}` ({data['prompt_count']} prompts)  ",
        f"**Hardware:** {data['hardware']}  ",
        "**Layer 2 model:** Qwen 2.5 Coder 7B (llama.cpp, Q4_K_M)  ",
        "",
        "---",
        "",
    ]

    if baseline and cold and warm:
        lines.extend(
            [
                "## Three-Way Comparison",
                "",
                "| Metric | Baseline | Isartor Cold | Isartor Warm |",
                "|--------|----------|--------------|--------------|",
                f"| Total requests | {baseline['total_requests']} | {cold['total_requests']} | {warm['total_requests']} |",
                f"| L3 (cloud) hits | {baseline['l3_hits']} (100%) | {cold['l3_hits']} ({cold['l3_hits']/cold['total_requests']*100:.1f}%) | {warm['l3_hits']} ({warm['l3_hits']/warm['total_requests']*100:.1f}%) |",
                f"| Deflection rate | 0% | {cold['deflection_rate']*100:.1f}% | {warm['deflection_rate']*100:.1f}% |",
                f"| P50 latency | {baseline['p50_ms']:.1f} ms | {cold['p50_ms']:.1f} ms | {warm['p50_ms']:.1f} ms |",
                f"| Est. cloud cost | ${baseline['total_cost_usd']:.4f} | ${cold['total_cost_usd']:.4f} | ${warm['total_cost_usd']:.4f} |",
                "",
            ]
        )

    def add_section(title: str, result: dict[str, Any] | None) -> None:
        if not result:
            return
        lines.extend(
            [
                f"## {title}",
                "",
                "| Layer | Hits | % of Traffic | Avg Latency (p50) |",
                "|-------|------|--------------|-------------------|",
                f"| L1a (exact) | {result['l1a_hits']} | {result['l1a_hits']/result['total_requests']*100:.1f}% | {result['l1a_p50_ms'] if result['l1a_p50_ms'] is not None else '-'} |",
                f"| L1b (semantic) | {result['l1b_hits']} | {result['l1b_hits']/result['total_requests']*100:.1f}% | {result['l1b_p50_ms'] if result['l1b_p50_ms'] is not None else '-'} |",
                f"| L2 (Qwen) | {result['l2_hits']} | {result['l2_hits']/result['total_requests']*100:.1f}% | {result['l2_p50_ms'] if result['l2_p50_ms'] is not None else '-'} |",
                f"| L3 (cloud) | {result['l3_hits']} | {result['l3_hits']/result['total_requests']*100:.1f}% | {result['l3_p50_ms'] if result['l3_p50_ms'] is not None else '-'} |",
                f"| **Total deflected** | **{result['l1a_hits'] + result['l1b_hits'] + result['l2_hits']}** | **{result['deflection_rate']*100:.1f}%** | |",
                f"| **Est. cost** | | | **${result['cost_per_req_usd']:.6f}/req** |",
                "",
                f"> Overall latency - P50: {result['p50_ms']:.1f} ms | P95: {result['p95_ms']:.1f} ms | P99: {result['p99_ms']:.1f} ms",
                ">",
                f"> Errors: {result['error_count']}",
                "",
            ]
        )

    add_section("Baseline - Without Isartor", baseline)
    add_section("Isartor Cold Cache", cold)
    add_section("Isartor Warm Cache", warm)
    return "\n".join(lines) + "\n"


def write_outputs(data: dict[str, Any], output_path: Path, report_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n")
    report_path.write_text(render_report(data))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Claude Code three-way benchmark harness")
    parser.add_argument("--three-way", action="store_true", help="Run baseline + cold + warm")
    parser.add_argument("--all-scenarios", action="store_true", help="Alias for --three-way")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic simulated responses")
    parser.add_argument("--scenario", choices=("baseline", "cold", "warm"), help="Run one scenario")
    parser.add_argument("--isartor-url", "--url", dest="isartor_url", default=os.environ.get("ISARTOR_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.environ.get("ISARTOR_API_KEY", ""))
    parser.add_argument("--direct-url", default=os.environ.get("DIRECT_LLM_URL", os.environ.get("ANTHROPIC_BASE_URL", COPILOT_COMPLETIONS_URL)))
    parser.add_argument("--direct-api-key", default=os.environ.get("COPILOT_KEY", os.environ.get("copilot_key", os.environ.get("DIRECT_LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", os.environ.get("ISARTOR_COPILOT_TOKEN", ""))))))
    parser.add_argument("--input", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--requests", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("ISARTOR_TIMEOUT", "120")))
    parser.add_argument("--output", default=str(RESULTS_DIR / "claude_code_benchmark.json"))
    parser.add_argument("--report", default=str(RESULTS_DIR / "claude_code_benchmark_report.md"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.all_scenarios:
        args.three_way = True
    if args.dry_run and not args.scenario:
        args.three_way = True
    if not args.three_way and not args.scenario:
        raise SystemExit("specify --three-way, --all-scenarios, or --scenario")

    fixture_path = Path(args.input)
    entries = load_fixture(fixture_path, args.requests)
    if not entries:
        raise SystemExit(f"fixture is empty: {fixture_path}")

    data: dict[str, Any] = {
        "benchmark": "claude_code_three_way",
        "timestamp": utc_now(),
        "hardware": hardware_summary(),
        "fixture": fixture_path.name,
        "prompt_count": len(entries),
        "mode": "dry-run" if args.dry_run else "live",
    }

    scenarios = ["baseline", "cold", "warm"] if args.three_way else [args.scenario]
    for scenario in scenarios:
        result = run_scenario(scenario, entries, args)
        data[SCENARIO_TO_KEY[scenario]] = result

    write_outputs(data, Path(args.output), Path(args.report))
    print(f"Wrote JSON report to {args.output}")
    print(f"Wrote Markdown report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

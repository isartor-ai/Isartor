#!/usr/bin/env python3
"""
Real Claude Code CLI benchmark: Baseline (passthrough) vs Isartor (full stack).

Architecture:
  Baseline: Claude CLI → Isartor(passthrough, no cache/L2) → Copilot GPT-5.4
  Isartor:  Claude CLI → Isartor(L1+L2+L3)                → Copilot GPT-5.4

Usage:
  python3 benchmarks/claude_bench.py \\
    --binary ./target/release/isartor \\
    --copilot-token ghp_xxx \\
    --model gpt-5.4 \\
    --output-dir results/claude_bench

  # Dry run (no API calls, no claude CLI required)
  python3 benchmarks/claude_bench.py --dry-run --output-dir /tmp/cb_test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

# ── Benchmark prompt ───────────────────────────────────────────────────────

BENCHMARK_PROMPT = """\
Build a complete TypeScript Express.js TODO API in this directory.

Requirements:
- package.json with express, typescript, ts-node, jest, ts-jest, and @types/*
- tsconfig.json (strict, ES2020, outDir: dist)
- src/types.ts with Todo interface (id, title, completed, createdAt)
- src/store.ts with in-memory Map store and CRUD helpers
- src/routes.ts with Express Router: GET/POST /todos, GET/PUT/DELETE /todos/:id
- src/app.ts Express app with JSON body parser and error handler
- src/index.ts entry point on port 3000
- jest.config.js for ts-jest
- tests/todos.test.ts covering create, read, update, delete, 404
- Dockerfile (multi-stage, node:20-alpine)
- .dockerignore

After creating all files, run: npm install && npx tsc --noEmit && npx jest
Fix any errors until the build and tests pass.\
"""

BASELINE_PORT = 8081
ISARTOR_PORT = 8082

# ── Data classes ───────────────────────────────────────────────────────────


@dataclass
class RunResult:
    scenario: str  # "baseline" or "isartor"
    workspace: str
    wall_time_s: float = 0.0
    claude_exit_code: int = -1
    claude_stdout: str = ""
    claude_stderr: str = ""
    # from claude --output-format json
    claude_cost_usd: float = 0.0
    claude_input_tokens: int = 0
    claude_output_tokens: int = 0
    claude_num_turns: int = 0
    claude_duration_ms: int = 0
    claude_session_id: str = ""
    # from isartor /debug/stats/prompts
    isartor_total_requests: int = 0
    isartor_l1a_hits: int = 0
    isartor_l1b_hits: int = 0
    isartor_l2_hits: int = 0
    isartor_l3_hits: int = 0
    isartor_deflection_rate: float = 0.0
    # validation
    files_created: list[str] = field(default_factory=list)
    npm_install_ok: bool = False
    tsc_ok: bool = False
    jest_ok: bool = False
    jest_tests_passed: int = 0
    jest_tests_total: int = 0


# ── HTTP helpers ───────────────────────────────────────────────────────────


def http_get_json(url: str, timeout: int = 10) -> dict | None:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def wait_for_health(port: int, timeout: int = 60) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── Isartor management ────────────────────────────────────────────────────


def start_isartor(
    binary: str,
    port: int,
    copilot_token: str,
    model: str,
    mode: str = "full",
    sidecar_url: str = "http://127.0.0.1:8090/v1",
) -> subprocess.Popen:
    """Start Isartor in passthrough (no cache/L2) or full (L1+L2+L3) mode."""
    env = os.environ.copy()
    env["ISARTOR__HOST_PORT"] = f"127.0.0.1:{port}"
    env["ISARTOR__LLM_PROVIDER"] = "copilot"
    env["ISARTOR__EXTERNAL_LLM_API_KEY"] = copilot_token
    env["ISARTOR__EXTERNAL_LLM_MODEL"] = model
    env["ISARTOR__EXTERNAL_LLM_URL"] = "https://api.githubcopilot.com/chat/completions"
    env["ISARTOR__GATEWAY_API_KEY"] = "benchmark-key"
    env["ISARTOR__OFFLINE_MODE"] = "false"
    env["ISARTOR__L3_TIMEOUT_SECS"] = "300"  # 5 min for large Claude Code payloads

    # Debug: show token info (without leaking it)
    tok_len = len(copilot_token) if copilot_token else 0
    tok_preview = (copilot_token[:4] + "..." + copilot_token[-4:]) if tok_len > 10 else "(empty/short)"
    print(f"  Token: {tok_preview} ({tok_len} chars)", file=sys.stderr)
    print(f"  Model: {model}, Provider: copilot", file=sys.stderr)
    print(f"  Port: {port}, Mode: {mode}", file=sys.stderr)

    if mode == "passthrough":
        env["ISARTOR__ENABLE_SLM_ROUTER"] = "false"
        env["ISARTOR__CACHE_MAX_CAPACITY"] = "0"
        env["ISARTOR__CACHE_TTL_SECS"] = "0"
    else:
        env["ISARTOR__ENABLE_SLM_ROUTER"] = "true"
        env["ISARTOR__LAYER2__SIDECAR_URL"] = sidecar_url
        env["ISARTOR__LAYER2__CLASSIFIER_MODE"] = "tiered"
        env["ISARTOR__LAYER2__MAX_ANSWER_TOKENS"] = "2048"

    proc = subprocess.Popen(
        [binary, "up"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_isartor(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def get_isartor_stats(port: int) -> dict:
    """Fetch stats from /debug/stats/prompts. Returns by_layer breakdown."""
    stats = http_get_json(f"http://127.0.0.1:{port}/debug/stats/prompts?limit=0")
    return stats or {}


def preflight_api_test(port: int, model: str = "gpt-4.1") -> bool:
    """Send a tiny /v1/messages request to verify Isartor→Copilot chain works."""
    url = f"http://127.0.0.1:{port}/v1/messages"
    payload = json.dumps({
        "model": model,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say hello"}],
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "benchmark-key",
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            text = ""
            for block in body.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            if text:
                print(f"  ✅ Preflight OK: \"{text[:80]}\"", file=sys.stderr)
                return True
            print(f"  ⚠ Preflight: empty response body: {json.dumps(body)[:300]}", file=sys.stderr)
            return False
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode()[:1000]
        except Exception:
            pass
        print(f"  ❌ Preflight HTTP {e.code}: {e.reason}", file=sys.stderr)
        if body_text:
            print(f"  Response body: {body_text}", file=sys.stderr)
        _dump_isartor_log()
        return False
    except Exception as e:
        print(f"  ❌ Preflight failed: {e}", file=sys.stderr)
        _dump_isartor_log()
        return False


def _dump_isartor_log() -> None:
    """Print tail of Isartor log for debugging."""
    for log_path in [
        Path.home() / ".isartor" / "isartor.log",
        Path("/tmp/isartor-bench.log"),
    ]:
        if log_path.exists():
            log_tail = log_path.read_text()[-2000:]
            print(f"  Isartor log ({log_path}, last 2000 chars):\n{log_tail}", file=sys.stderr)
            return
    print("  (no Isartor log found)", file=sys.stderr)


# ── Claude CLI ─────────────────────────────────────────────────────────────


def run_copilot_cli(
    workspace: Path,
    prompt: str,
    copilot_token: str,
    port: int | None = None,
    timeout: int = 900,
    max_turns: int = 50,
) -> tuple[int, str, str]:
    """Run `copilot -p` in workspace. Returns (exit_code, stdout, stderr).

    If port is None, Copilot CLI talks directly to GitHub (baseline).
    If port is set, routes through Isartor on that port.
    """
    env = os.environ.copy()
    env["GH_TOKEN"] = copilot_token
    env["GITHUB_TOKEN"] = copilot_token

    if port is not None:
        # Route through Isartor — set as custom provider endpoint
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["OPENAI_API_KEY"] = "benchmark-key"

    cmd = [
        "copilot",
        "-p", prompt,
        "--allow-all-tools",
        "--output-format", "json",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"copilot CLI timed out after {timeout}s"


def run_claude_cli(
    workspace: Path,
    port: int,
    model: str,
    prompt: str,
    timeout: int = 900,
    max_turns: int = 50,
) -> tuple[int, str, str]:
    """Run `claude -p` in workspace. Returns (exit_code, stdout, stderr)."""
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["ANTHROPIC_AUTH_TOKEN"] = "benchmark-key"
    env["ANTHROPIC_API_KEY"] = "benchmark-key"
    env["ANTHROPIC_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["DISABLE_NON_ESSENTIAL_MODEL_CALLS"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "16000"
    env["CLAUDE_CODE_SKIP_UPDATE_CHECK"] = "1"

    cmd = [
        "claude",
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--max-turns", str(max_turns),
        "--model", model,
        "--verbose",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"claude CLI timed out after {timeout}s"


def parse_claude_json(stdout: str) -> dict:
    """Parse the JSON output from `claude --output-format json`.

    Claude CLI may return a single JSON object or a JSON array of
    message objects.  When it returns an array, the last element with
    type "result" carries the summary we need.
    """
    stdout = stdout.strip()
    if not stdout:
        return {}

    parsed: Any = None
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        # Try last line (claude may print non-JSON before the result)
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith(("{", "[")):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    if parsed is None:
        return {}

    # If it's already a dict, return as-is
    if isinstance(parsed, dict):
        return parsed

    # If it's a list, find the last "result" entry (or last dict)
    if isinstance(parsed, list):
        result_entry: dict = {}
        for item in reversed(parsed):
            if isinstance(item, dict):
                if item.get("type") == "result":
                    return item
                if not result_entry:
                    result_entry = item
        return result_entry

    return {}


# ── Workspace management ──────────────────────────────────────────────────


def init_workspace(base: Path, name: str) -> Path:
    ws = base / name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=ws, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "initial"],
        cwd=ws, capture_output=True,
    )
    return ws


def commit_workspace(ws: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", message],
        cwd=ws, capture_output=True,
    )


def push_workspace(ws: Path, repo_url: str, branch: str) -> bool:
    # Don't push empty workspaces
    files = [f for f in ws.rglob("*") if f.is_file() and ".git" not in f.parts]
    if not files:
        print(f"  ⚠ Skipping push — workspace is empty", file=sys.stderr)
        return False
    subprocess.run(
        ["git", "remote", "remove", "dist"],
        cwd=ws, capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "dist", repo_url],
        cwd=ws, capture_output=True,
    )
    result = subprocess.run(
        ["git", "push", "--force", "dist", f"HEAD:refs/heads/{branch}"],
        cwd=ws, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  ✅ Pushed to {branch}", file=sys.stderr)
        return True
    print(f"  ❌ Push failed: {result.stderr[:300]}", file=sys.stderr)
    return False


def list_files(ws: Path) -> list[str]:
    files = []
    for f in ws.rglob("*"):
        if f.is_file() and ".git" not in f.parts and "node_modules" not in f.parts:
            files.append(str(f.relative_to(ws)))
    return sorted(files)


# ── Validation ────────────────────────────────────────────────────────────


def validate_workspace(ws: Path) -> dict:
    """Run npm install, tsc, jest on the workspace."""
    result: dict[str, Any] = {
        "npm_install_ok": False,
        "tsc_ok": False,
        "jest_ok": False,
        "jest_tests_passed": 0,
        "jest_tests_total": 0,
    }

    if not (ws / "package.json").exists():
        print("    ⚠ No package.json found", file=sys.stderr)
        return result

    # npm install
    r = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=ws, capture_output=True, text=True, timeout=120,
    )
    result["npm_install_ok"] = r.returncode == 0
    if not result["npm_install_ok"]:
        print(f"    ⚠ npm install failed: {r.stderr[:200]}", file=sys.stderr)
        return result

    # tsc --noEmit
    if (ws / "tsconfig.json").exists():
        r = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            cwd=ws, capture_output=True, text=True, timeout=60,
        )
        result["tsc_ok"] = r.returncode == 0
        if not result["tsc_ok"]:
            print(f"    ⚠ tsc failed: {r.stdout[:200]}", file=sys.stderr)

    # jest
    r = subprocess.run(
        ["npx", "jest", "--json", "--forceExit"],
        cwd=ws, capture_output=True, text=True, timeout=120,
    )
    result["jest_ok"] = r.returncode == 0
    try:
        jest_out = json.loads(r.stdout)
        result["jest_tests_passed"] = jest_out.get("numPassedTests", 0)
        result["jest_tests_total"] = jest_out.get("numTotalTests", 0)
    except (json.JSONDecodeError, KeyError):
        pass

    return result


# ── Run a scenario ────────────────────────────────────────────────────────


def run_scenario(
    scenario: str,
    binary: str,
    port: int,
    copilot_token: str,
    model: str,
    workspace: Path,
    mode: str,
    sidecar_url: str,
    timeout: int,
    max_turns: int,
    dry_run: bool = False,
    cli: str = "copilot",  # "copilot" or "claude"
) -> RunResult:
    res = RunResult(scenario=scenario, workspace=str(workspace))

    if dry_run:
        print(f"\n  [DRY RUN] {scenario}: would run `{cli} -p` in {workspace}", file=sys.stderr)
        res.wall_time_s = 0.1
        res.claude_exit_code = 0
        res.files_created = ["(dry-run)"]
        return res

    # Copilot CLI baseline: NO Isartor needed — talks directly to GitHub
    if cli == "copilot" and mode == "passthrough":
        print(f"  Running `copilot -p` directly (no Isartor)...", file=sys.stderr)
        t0 = time.time()
        exit_code, stdout, stderr = run_copilot_cli(
            workspace, BENCHMARK_PROMPT, copilot_token,
            port=None, timeout=timeout, max_turns=max_turns,
        )
        wall_time = time.time() - t0
        res.wall_time_s = wall_time
        res.claude_exit_code = exit_code
        res.claude_stdout = stdout[:5000]
        res.claude_stderr = stderr[:2000]
        _parse_cli_output(res, stdout, stderr, exit_code)
        # No Isartor stats for direct baseline
        commit_workspace(workspace, f"copilot-bench: {scenario}")
        res.files_created = list_files(workspace)
        print(f"  Files created: {len(res.files_created)}", file=sys.stderr)
        print(f"  Validating workspace...", file=sys.stderr)
        _apply_validation(res, validate_workspace(workspace))
        commit_workspace(workspace, f"post-validation: {scenario}")
        return res

    # All other modes: start Isartor
    print(f"  Starting Isartor ({mode}) on port {port}...", file=sys.stderr)
    proc = start_isartor(binary, port, copilot_token, model, mode, sidecar_url)
    if not wait_for_health(port, timeout=60):
        stop_isartor(proc)
        print(f"  ❌ Isartor failed to start on port {port}", file=sys.stderr)
        _, proc_stderr = proc.communicate(timeout=5)
        if proc_stderr:
            print(f"  Isartor stderr: {proc_stderr.decode()[:500]}", file=sys.stderr)
        res.claude_exit_code = -2
        return res
    print(f"  ✅ Isartor ready on port {port}", file=sys.stderr)

    # Pre-flight
    print(f"  Running preflight API test...", file=sys.stderr)
    if not preflight_api_test(port, model):
        stop_isartor(proc)
        print(f"  ❌ Preflight failed — Isartor→Copilot chain is broken", file=sys.stderr)
        res.claude_exit_code = -3
        return res

    # Run CLI
    print(f"  Running `{cli} -p` (max {max_turns} turns, timeout {timeout}s)...", file=sys.stderr)
    t0 = time.time()
    if cli == "copilot":
        exit_code, stdout, stderr = run_copilot_cli(
            workspace, BENCHMARK_PROMPT, copilot_token,
            port=port, timeout=timeout, max_turns=max_turns,
        )
    else:
        exit_code, stdout, stderr = run_claude_cli(
            workspace, port, model, BENCHMARK_PROMPT, timeout, max_turns,
        )
    wall_time = time.time() - t0

    res.wall_time_s = wall_time
    res.claude_exit_code = exit_code
    res.claude_stdout = stdout[:5000]
    res.claude_stderr = stderr[:2000]

    _parse_cli_output(res, stdout, stderr, exit_code)

    # Get Isartor stats
    stats = get_isartor_stats(port)
    if stats:
        by_layer = stats.get("by_layer", {})
        res.isartor_total_requests = stats.get("total_prompts", 0)
        res.isartor_l1a_hits = by_layer.get("l1a", 0)
        res.isartor_l1b_hits = by_layer.get("l1b", 0)
        res.isartor_l2_hits = by_layer.get("l2", 0)
        res.isartor_l3_hits = by_layer.get("l3", 0)
        total = res.isartor_total_requests or 1
        deflected = res.isartor_l1a_hits + res.isartor_l1b_hits + res.isartor_l2_hits
        res.isartor_deflection_rate = deflected / total * 100

    stop_isartor(proc)
    commit_workspace(workspace, f"{cli}-bench: {scenario}")
    res.files_created = list_files(workspace)
    print(f"  Files created: {len(res.files_created)}", file=sys.stderr)
    for f in res.files_created[:15]:
        print(f"    {f}", file=sys.stderr)
    if len(res.files_created) > 15:
        print(f"    ... and {len(res.files_created) - 15} more", file=sys.stderr)

    print(f"  Validating workspace...", file=sys.stderr)
    _apply_validation(res, validate_workspace(workspace))
    commit_workspace(workspace, f"post-validation: {scenario}")
    return res


def _parse_cli_output(res: RunResult, stdout: str, stderr: str, exit_code: int) -> None:
    """Parse Claude/Copilot CLI output and populate result fields."""
    if exit_code != 0:
        print(f"  ⚠ CLI exited with code {exit_code}", file=sys.stderr)
        if stderr:
            print(f"  stderr (last 1000 chars):\n{stderr[-1000:]}", file=sys.stderr)
        if stdout:
            print(f"  stdout (last 500 chars):\n{stdout[-500:]}", file=sys.stderr)
        _dump_isartor_log()

    parsed = parse_claude_json(stdout)
    if parsed:
        res.claude_cost_usd = parsed.get("cost_usd", 0.0) or parsed.get("total_cost_usd", 0.0) or 0.0
        res.claude_num_turns = parsed.get("num_turns", 0) or 0
        res.claude_duration_ms = parsed.get("duration_ms", 0) or 0
        res.claude_session_id = parsed.get("session_id", "") or ""
        usage = parsed.get("usage", {}) or {}
        res.claude_input_tokens = usage.get("input_tokens", 0) or 0
        res.claude_output_tokens = usage.get("output_tokens", 0) or 0


def _apply_validation(res: RunResult, val: dict) -> None:
    res.npm_install_ok = val["npm_install_ok"]
    res.tsc_ok = val["tsc_ok"]
    res.jest_ok = val["jest_ok"]
    res.jest_tests_passed = val["jest_tests_passed"]
    res.jest_tests_total = val["jest_tests_total"]


# ── Report generation ─────────────────────────────────────────────────────


def fmt(r: RunResult | None, attr: str, template: str = "{}") -> str:
    if r is None:
        return "—"
    val = getattr(r, attr, None)
    if val is None:
        return "—"
    return template.format(val)


def generate_report(
    baseline: RunResult | None,
    isartor: RunResult | None,
    output_dir: Path,
) -> None:
    # ── tokens.json ──
    tokens: dict[str, Any] = {}
    for label, r in [("baseline", baseline), ("isartor", isartor)]:
        if r:
            tokens[label] = {
                "wall_time_s": round(r.wall_time_s, 1),
                "claude": {
                    "input_tokens": r.claude_input_tokens,
                    "output_tokens": r.claude_output_tokens,
                    "cost_usd": r.claude_cost_usd,
                    "num_turns": r.claude_num_turns,
                    "duration_ms": r.claude_duration_ms,
                },
                "isartor": {
                    "total_requests": r.isartor_total_requests,
                    "l1a_hits": r.isartor_l1a_hits,
                    "l1b_hits": r.isartor_l1b_hits,
                    "l2_hits": r.isartor_l2_hits,
                    "l3_hits": r.isartor_l3_hits,
                    "deflection_rate": round(r.isartor_deflection_rate, 1),
                },
                "validation": {
                    "npm_install": r.npm_install_ok,
                    "tsc_compile": r.tsc_ok,
                    "jest_pass": r.jest_ok,
                    "tests_passed": r.jest_tests_passed,
                    "tests_total": r.jest_tests_total,
                },
                "files_created": len(r.files_created),
            }

    if baseline and isartor:
        bl_cloud = baseline.isartor_l3_hits
        is_cloud = isartor.isartor_l3_hits
        saved = bl_cloud - is_cloud if bl_cloud > 0 else 0
        tokens["comparison"] = {
            "cloud_calls_baseline": bl_cloud,
            "cloud_calls_isartor": is_cloud,
            "cloud_calls_saved": saved,
            "cloud_call_reduction_pct": round(saved / max(bl_cloud, 1) * 100, 1),
            "wall_time_saved_s": round(baseline.wall_time_s - isartor.wall_time_s, 1),
        }

    (output_dir / "tokens.json").write_text(json.dumps(tokens, indent=2) + "\n")
    print(f"  Written: tokens.json", file=sys.stderr)

    # ── code.diff ──
    if baseline and isartor:
        diff = subprocess.run(
            ["diff", "-ruN",
             "--exclude=node_modules", "--exclude=.git",
             str(baseline.workspace), str(isartor.workspace)],
            capture_output=True, text=True,
        )
        (output_dir / "code.diff").write_text(diff.stdout or "(identical)\n")
        print(f"  Written: code.diff", file=sys.stderr)

    # ── summary.md ──
    lines = [
        "# Code Generation Benchmark: Baseline vs Isartor",
        "",
        "Autonomous CLI building a TypeScript Express TODO API.",
        "",
        "- **Baseline**: Direct Copilot (no Isartor)",
        "- **Isartor**: Full stack (L1 cache + L2 SLM + L3 cloud)",
        "",
        "| Metric | Baseline | Isartor |",
        "|--------|----------|---------|",
        f"| Wall time | {fmt(baseline, 'wall_time_s', '{:.1f}s')} | {fmt(isartor, 'wall_time_s', '{:.1f}s')} |",
        f"| Claude turns | {fmt(baseline, 'claude_num_turns')} | {fmt(isartor, 'claude_num_turns')} |",
        f"| Input tokens | {fmt(baseline, 'claude_input_tokens', '{:,}')} | {fmt(isartor, 'claude_input_tokens', '{:,}')} |",
        f"| Output tokens | {fmt(baseline, 'claude_output_tokens', '{:,}')} | {fmt(isartor, 'claude_output_tokens', '{:,}')} |",
        f"| Cost (USD) | {fmt(baseline, 'claude_cost_usd', '${:.4f}')} | {fmt(isartor, 'claude_cost_usd', '${:.4f}')} |",
        f"| API calls | {fmt(baseline, 'isartor_total_requests')} | {fmt(isartor, 'isartor_total_requests')} |",
        f"| L1a exact cache | {fmt(baseline, 'isartor_l1a_hits')} | {fmt(isartor, 'isartor_l1a_hits')} |",
        f"| L1b semantic cache | {fmt(baseline, 'isartor_l1b_hits')} | {fmt(isartor, 'isartor_l1b_hits')} |",
        f"| L2 SLM answers | {fmt(baseline, 'isartor_l2_hits')} | {fmt(isartor, 'isartor_l2_hits')} |",
        f"| L3 cloud calls | {fmt(baseline, 'isartor_l3_hits')} | {fmt(isartor, 'isartor_l3_hits')} |",
        f"| Deflection rate | {fmt(baseline, 'isartor_deflection_rate', '{:.1f}%')} | {fmt(isartor, 'isartor_deflection_rate', '{:.1f}%')} |",
        f"| npm install | {'✅' if baseline and baseline.npm_install_ok else '—'} | {'✅' if isartor and isartor.npm_install_ok else '—'} |",
        f"| tsc compile | {'✅' if baseline and baseline.tsc_ok else '—'} | {'✅' if isartor and isartor.tsc_ok else '—'} |",
        f"| Jest tests | {'✅' if baseline and baseline.jest_ok else '—'} | {'✅' if isartor and isartor.jest_ok else '—'} |",
        f"| Tests passed | {fmt(baseline, 'jest_tests_passed')}/{fmt(baseline, 'jest_tests_total')} | {fmt(isartor, 'jest_tests_passed')}/{fmt(isartor, 'jest_tests_total')} |",
        f"| Files created | {len(baseline.files_created) if baseline else '—'} | {len(isartor.files_created) if isartor else '—'} |",
    ]

    if baseline and isartor:
        bl_cloud = baseline.isartor_l3_hits
        is_cloud = isartor.isartor_l3_hits
        saved = bl_cloud - is_cloud
        pct = saved / max(bl_cloud, 1) * 100
        dt = baseline.wall_time_s - isartor.wall_time_s
        lines += [
            "",
            "## Savings",
            "",
            f"- **Cloud calls**: {bl_cloud} → {is_cloud} "
            f"(**{saved} saved, {pct:.1f}% reduction**)",
            f"- **Wall time**: {baseline.wall_time_s:.1f}s → {isartor.wall_time_s:.1f}s "
            f"(**{dt:+.1f}s**)",
        ]

    lines += [
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))
    print(f"  Written: summary.md", file=sys.stderr)


# ── CLI & main ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Code generation benchmark: Baseline vs Isartor",
    )
    p.add_argument(
        "--cli", default="copilot", choices=["copilot", "claude"],
        help="Which CLI to use (default: copilot)",
    )
    p.add_argument(
        "--binary", default="./target/release/isartor",
        help="Path to Isartor binary",
    )
    p.add_argument(
        "--copilot-token",
        default=os.environ.get("COPILOT_WORKFLOW_KEY", ""),
        help="GitHub Copilot token (or set COPILOT_WORKFLOW_KEY)",
    )
    p.add_argument("--model", default="gpt-4.1", help="LLM model name")
    p.add_argument(
        "--output-dir", default="benchmarks/results/claude_bench",
        help="Output directory for results",
    )
    p.add_argument(
        "--sidecar-url", default="http://127.0.0.1:8090/v1",
        help="L2 SLM sidecar URL",
    )
    p.add_argument(
        "--timeout", type=int, default=900,
        help="CLI timeout in seconds (default 900 = 15 min)",
    )
    p.add_argument(
        "--max-turns", type=int, default=50,
        help="Max agentic turns",
    )
    p.add_argument("--push-repo", default="", help="Git remote URL to push workspaces")
    p.add_argument("--dry-run", action="store_true", help="Simulate without API calls")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--isartor-only", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ws_base = output_dir / "workspaces"
    cli = args.cli

    if not args.dry_run and not args.copilot_token:
        print("❌ --copilot-token required (or set COPILOT_WORKFLOW_KEY)", file=sys.stderr)
        return 1

    # Check CLI availability
    if not args.dry_run:
        cli_cmd = cli
        r = subprocess.run([cli_cmd, "--version"], capture_output=True, text=True)
        if r.returncode != 0:
            pkg = "@github/copilot" if cli == "copilot" else "@anthropic-ai/claude-code"
            print(f"❌ `{cli_cmd}` CLI not found. Install: npm install -g {pkg}", file=sys.stderr)
            return 1
        print(f"CLI: {cli_cmd} {r.stdout.strip()}", file=sys.stderr)

    print(f"Runner: {cli}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Binary: {args.binary}", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)
    print(f"Max turns: {args.max_turns}, Timeout: {args.timeout}s", file=sys.stderr)

    baseline_result: RunResult | None = None
    isartor_result: RunResult | None = None

    # ── Baseline ──
    if not args.isartor_only:
        if cli == "copilot":
            print("\n═══ Baseline (direct Copilot — no Isartor) ═══", file=sys.stderr)
        else:
            print("\n═══ Baseline (passthrough — no cache, no L2) ═══", file=sys.stderr)
        ws = init_workspace(ws_base, "baseline")
        baseline_result = run_scenario(
            scenario="baseline",
            binary=args.binary,
            port=BASELINE_PORT,
            copilot_token=args.copilot_token,
            model=args.model,
            workspace=ws,
            mode="passthrough",
            sidecar_url=args.sidecar_url,
            timeout=args.timeout,
            max_turns=args.max_turns,
            dry_run=args.dry_run,
            cli=cli,
        )
        print(
            f"\n  Result: {baseline_result.wall_time_s:.1f}s, "
            f"exit={baseline_result.claude_exit_code}, "
            f"files={len(baseline_result.files_created)}, "
            f"npm={'✅' if baseline_result.npm_install_ok else '❌'}, "
            f"tsc={'✅' if baseline_result.tsc_ok else '❌'}, "
            f"jest={baseline_result.jest_tests_passed}/{baseline_result.jest_tests_total}",
            file=sys.stderr,
        )

    # ── Isartor ──
    if not args.baseline_only:
        print("\n═══ Isartor (L1 cache + L2 SLM + L3 cloud) ═══", file=sys.stderr)
        ws = init_workspace(ws_base, "isartor")
        isartor_result = run_scenario(
            scenario="isartor",
            binary=args.binary,
            port=ISARTOR_PORT,
            copilot_token=args.copilot_token,
            model=args.model,
            workspace=ws,
            mode="full",
            sidecar_url=args.sidecar_url,
            timeout=args.timeout,
            max_turns=args.max_turns,
            dry_run=args.dry_run,
            cli=cli,
        )
        print(
            f"\n  Result: {isartor_result.wall_time_s:.1f}s, "
            f"exit={isartor_result.claude_exit_code}, "
            f"files={len(isartor_result.files_created)}, "
            f"deflection={isartor_result.isartor_deflection_rate:.1f}%, "
            f"npm={'✅' if isartor_result.npm_install_ok else '❌'}, "
            f"tsc={'✅' if isartor_result.tsc_ok else '❌'}, "
            f"jest={isartor_result.jest_tests_passed}/{isartor_result.jest_tests_total}",
            file=sys.stderr,
        )

    # ── Generate report ──
    print("\n═══ Generating report ═══", file=sys.stderr)
    generate_report(baseline_result, isartor_result, output_dir)

    # ── Full results JSON ──
    full: dict[str, Any] = {}
    for label, r in [("baseline", baseline_result), ("isartor", isartor_result)]:
        if r:
            d = asdict(r)
            d.pop("claude_stdout", None)
            d.pop("claude_stderr", None)
            full[label] = d
    (output_dir / "full_results.json").write_text(json.dumps(full, indent=2) + "\n")

    # ── Push workspaces ──
    if args.push_repo:
        print("\n═══ Pushing workspaces ═══", file=sys.stderr)
        ts = time.strftime("%Y%m%d-%H%M", time.gmtime())
        if baseline_result and baseline_result.workspace != "":
            push_workspace(
                Path(baseline_result.workspace),
                args.push_repo,
                f"benchmark/baseline-{ts}",
            )
        if isartor_result and isartor_result.workspace != "":
            push_workspace(
                Path(isartor_result.workspace),
                args.push_repo,
                f"benchmark/isartor-{ts}",
            )

    print(f"\n✅ All outputs in {output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
scenario_runner.py — Code-generation benchmark for Isartor.

Replays claude_code_todo_app.jsonl through two scenarios:
  A) Claude Code + Copilot  (baseline, no Isartor)
  B) Claude Code + Isartor + Copilot

For each scenario the runner:
  1. Creates a fresh workspace directory
  2. Sends each fixture prompt to the target endpoint
  3. Writes the LLM response into the workspace as code files
  4. Commits results into a local git repo
  5. Runs validate_todo_app.py checks
  6. Produces tokens.json  (per-prompt and aggregate token accounting)
  7. Produces code.diff    (unified diff between the two workspaces)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AVG_INPUT_TOKENS = 800
AVG_OUTPUT_TOKENS = 200
RETRYABLE_HTTP_STATUSES = {429, 502, 503, 504}
MAX_HTTP_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_SECS = 1.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PromptResult:
    step: int
    phase: str
    prompt_preview: str
    layer: str
    latency_ms: float
    input_tokens_est: int
    output_tokens_est: int
    status: int
    error: str = ""
    files_written: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    scenario: str
    workspace: str
    total_prompts: int
    total_input_tokens: int
    total_output_tokens: int
    total_cloud_input_tokens: int
    total_cloud_output_tokens: int
    deflection_rate: float
    l1a_hits: int
    l1b_hits: int
    l2_hits: int
    l3_hits: int
    errors: int
    validation_passed: bool
    validation_total: int
    validation_failed: int
    prompts: list[PromptResult]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def send_anthropic_request(
    url: str,
    api_key: str,
    prompt: str,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    """POST an Anthropic-compatible chat request. Returns (status, headers, body)."""
    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": "gpt-5.4",
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": api_key,
    }
    if extra_headers:
        headers.update(extra_headers)

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                resp_body = resp.read().decode()
                return resp.status, resp_headers, resp_body
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE_HTTP_STATUSES and attempt < MAX_HTTP_ATTEMPTS:
                wait = HTTP_RETRY_BACKOFF_SECS * (2 ** (attempt - 1))
                print(f"  [retry] HTTP {e.code} on attempt {attempt}/{MAX_HTTP_ATTEMPTS}; "
                      f"retrying in {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            return e.code, {}, e.read().decode() if hasattr(e, "read") else ""
        except Exception as e:
            if attempt < MAX_HTTP_ATTEMPTS:
                time.sleep(HTTP_RETRY_BACKOFF_SECS)
                continue
            return 0, {}, str(e)

    return 0, {}, "max retries exhausted"


def send_copilot_request(
    prompt: str,
    github_token: str,
    timeout: float,
    model: str = "gpt-5.4",
    _session_cache: dict[str, str] = {},
) -> tuple[int, dict[str, str], str]:
    """Direct Copilot request (baseline, no Isartor)."""
    import urllib.request
    import urllib.error

    # Cache the session token to avoid exchanging on every request
    if "token" not in _session_cache or "expires_at" not in _session_cache \
       or time.time() > _session_cache.get("expires_at", 0) - 60:
        tok_req = urllib.request.Request(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/json",
                "User-Agent": "GitHubCopilotChat/0.29.1",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(tok_req, timeout=30) as resp:
                session = json.loads(resp.read().decode())
                _session_cache["token"] = session.get("token", "")
                _session_cache["expires_at"] = session.get("expires_at", time.time() + 600)
        except Exception as e:
            return 0, {}, f"copilot token exchange failed: {e}"

    session_token = _session_cache["token"]

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {session_token}",
        "Editor-Version": "vscode/1.99.0",
        "Copilot-Integration-Id": "vscode-chat",
        "User-Agent": "GitHubCopilotChat/0.29.1",
    }

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        req = urllib.request.Request(
            "https://api.githubcopilot.com/chat/completions",
            data=body, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                return resp.status, resp_headers, resp.read().decode()
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE_HTTP_STATUSES and attempt < MAX_HTTP_ATTEMPTS:
                time.sleep(HTTP_RETRY_BACKOFF_SECS * (2 ** (attempt - 1)))
                continue
            return e.code, {}, e.read().decode() if hasattr(e, "read") else ""
        except Exception as e:
            if attempt < MAX_HTTP_ATTEMPTS:
                time.sleep(HTTP_RETRY_BACKOFF_SECS)
                continue
            return 0, {}, str(e)

    return 0, {}, "max retries exhausted"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_response_text(body: str) -> str:
    """Extract the assistant's text from an Anthropic or OpenAI response."""
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return body

    # Anthropic Messages format
    if "content" in obj and isinstance(obj["content"], list):
        parts = []
        for block in obj["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        if parts:
            return "\n".join(parts)

    # OpenAI Chat Completions format
    if "choices" in obj:
        for choice in obj["choices"]:
            msg = choice.get("message", {})
            if msg.get("content"):
                return msg["content"]

    # Native Isartor format
    if "message" in obj:
        return obj["message"]

    return body


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract fenced code blocks as [(filename_hint, code), ...].

    If the block has a preceding line like `**src/types.ts**` or
    `// filename: src/types.ts` we use that as the filename hint.
    """
    pattern = re.compile(
        r'(?:(?:^|\n)(?:\*\*|`)?([a-zA-Z0-9_./-]+\.[a-z]+)(?:\*\*|`)?[ \t]*\n)?'
        r'```[a-z]*\n(.*?)```',
        re.DOTALL,
    )
    results = []
    for m in pattern.finditer(text):
        fname = m.group(1) or ""
        code = m.group(2)
        results.append((fname.strip(), code))
    if not results:
        # No fenced blocks — treat the whole text as a single file
        results.append(("", text))
    return results


def guess_filename(prompt: str, hint: str, step: int) -> str:
    """Derive a filesystem path from the prompt and code-block hint."""
    if hint:
        return hint

    prompt_lower = prompt.lower()

    mapping = [
        ("package.json", "package.json"),
        ("tsconfig.json", "tsconfig.json"),
        ("jest.config", "jest.config.js"),
        (".gitignore", ".gitignore"),
        (".env.example", ".env.example"),
        ("dockerfile", "Dockerfile"),
        ("docker-compose", "docker-compose.yml"),
        ("readme.md", "README.md"),
        ("src/types.ts", "src/types.ts"),
        ("src/store.ts", "src/store.ts"),
        ("src/app.ts", "src/app.ts"),
        ("src/server.ts", "src/server.ts"),
        ("src/routes/todos.ts", "src/routes/todos.ts"),
        ("src/middleware/errorhandler", "src/middleware/errorHandler.ts"),
        ("src/middleware/notfound", "src/middleware/notFound.ts"),
        ("public/index.html", "public/index.html"),
        ("public/app.js", "public/app.js"),
        ("public/styles.css", "public/styles.css"),
        ("__tests__/unit/store.test", "__tests__/unit/store.test.ts"),
        ("__tests__/integration/todos.test", "__tests__/integration/todos.test.ts"),
    ]
    for keyword, path in mapping:
        if keyword.lower() in prompt_lower:
            return path

    return f"output/step_{step:03d}.txt"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------

def init_workspace(base_dir: Path, name: str) -> Path:
    """Create a fresh git-initialised workspace."""
    ws = base_dir / name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "initial (empty)"],
        cwd=ws, check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "benchmark", "GIT_AUTHOR_EMAIL": "bench@isartor",
             "GIT_COMMITTER_NAME": "benchmark", "GIT_COMMITTER_EMAIL": "bench@isartor"},
    )
    return ws


def write_file(ws: Path, relpath: str, content: str) -> None:
    """Write a file into the workspace, creating parent dirs."""
    target = ws / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def commit_workspace(ws: Path, message: str) -> None:
    """Stage all files and commit."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "benchmark",
        "GIT_AUTHOR_EMAIL": "bench@isartor",
        "GIT_COMMITTER_NAME": "benchmark",
        "GIT_COMMITTER_EMAIL": "bench@isartor",
    }
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ws,
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=ws, check=True, env=env,
        )


def push_workspace(ws: Path, repo_url: str, branch: str) -> bool:
    """Push workspace contents to a remote git repo. Returns True on success."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "isartor-benchmark",
        "GIT_AUTHOR_EMAIL": "bench@isartor.ai",
        "GIT_COMMITTER_NAME": "isartor-benchmark",
        "GIT_COMMITTER_EMAIL": "bench@isartor.ai",
    }
    # Add remote if not already set
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=ws, capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=ws, check=True)
    else:
        subprocess.run(["git", "remote", "set-url", "origin", repo_url], cwd=ws, check=True)

    # Force push to branch (overwrites previous benchmark results)
    result = subprocess.run(
        ["git", "push", "--force", "origin", f"HEAD:{branch}"],
        cwd=ws, capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        print(f"  ✅ Pushed {ws.name} → {repo_url} branch={branch}", file=sys.stderr)
        return True
    print(f"  ❌ Push failed: {result.stderr.strip()[:300]}", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------

SAMPLE_RESPONSES: dict[str, str] = {
    "package.json": '```json\n{"name":"todo-app","version":"1.0.0","scripts":{"build":"tsc","start":"node dist/server.js","dev":"ts-node-dev --respawn src/server.ts","test":"jest --runInBand"},"dependencies":{"express":"^4.18.2","cors":"^2.8.5","uuid":"^9.0.0"},"devDependencies":{"typescript":"^5.3.3"}}\n```',
    "tsconfig.json": '```json\n{"compilerOptions":{"target":"ES2020","module":"commonjs","outDir":"./dist","rootDir":"./src","strict":true,"esModuleInterop":true}}\n```',
    "src/types.ts": '```typescript\nexport interface Todo {\n  id: string;\n  title: string;\n  completed: boolean;\n  createdAt: string;\n  updatedAt: string;\n}\nexport type CreateTodoInput = { title: string; completed?: boolean };\nexport type UpdateTodoInput = { title?: string; completed?: boolean };\n```',
}


def dry_run_response(prompt: str, step: int) -> str:
    """Return a synthetic response for dry-run mode."""
    for key, resp in SAMPLE_RESPONSES.items():
        if key.lower() in prompt.lower():
            return resp
    return f"```\n// Step {step}: generated code placeholder\nconsole.log('todo');\n```"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    entries: list[dict[str, Any]],
    workspace: Path,
    args: argparse.Namespace,
) -> ScenarioResult:
    """Replay all fixture prompts against one target, writing code into workspace."""

    results: list[PromptResult] = []
    layer_counts = {"l1a": 0, "l1b": 0, "l2": 0, "l3": 0}

    is_baseline = name == "baseline"

    for i, entry in enumerate(entries):
        step = entry.get("step", i + 1)
        phase = entry.get("phase", "standalone")
        prompt = entry["prompt"]
        prompt_preview = prompt[:80]

        print(f"  [{name}] step {step}/{len(entries)}: {prompt_preview}...", file=sys.stderr)

        t0 = time.monotonic()

        if args.dry_run:
            response_text = dry_run_response(prompt, step)
            status = 200
            headers: dict[str, str] = {"x-isartor-layer": "l2"} if not is_baseline else {}
            latency_ms = 50.0 + (step * 3)
        elif is_baseline:
            if args.copilot_token:
                status, headers, body = send_copilot_request(
                    prompt, args.copilot_token, args.timeout,
                    model=args.model,
                )
            else:
                status, headers, body = send_anthropic_request(
                    args.direct_url, args.direct_api_key, prompt, args.timeout,
                )
            latency_ms = (time.monotonic() - t0) * 1000
            response_text = extract_response_text(body) if status == 200 else ""
        else:
            status, headers, body = send_anthropic_request(
                f"{args.isartor_url}/v1/messages",
                args.api_key,
                prompt,
                args.timeout,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            response_text = extract_response_text(body) if status == 200 else ""

        # Determine layer
        layer_raw = headers.get("x-isartor-layer", "l3") if not is_baseline else "l3"
        layer = layer_raw if layer_raw in layer_counts else "l3"
        layer_counts[layer] += 1

        # Write code into workspace
        files_written: list[str] = []
        if status == 200 and response_text:
            blocks = extract_code_blocks(response_text)
            for block_idx, (hint, code) in enumerate(blocks):
                fname = guess_filename(prompt, hint, step)
                if block_idx > 0 and not hint:
                    base, ext = os.path.splitext(fname)
                    fname = f"{base}_{block_idx}{ext}"
                write_file(workspace, fname, code)
                files_written.append(fname)

        input_tok = estimate_tokens(prompt)
        output_tok = estimate_tokens(response_text) if response_text else 0

        results.append(PromptResult(
            step=step,
            phase=phase,
            prompt_preview=prompt_preview,
            layer=layer,
            latency_ms=latency_ms,
            input_tokens_est=input_tok,
            output_tokens_est=output_tok,
            status=status,
            error="" if status == 200 else f"HTTP {status}",
            files_written=files_written,
        ))

    # Commit all generated code
    commit_workspace(workspace, f"Generated code from {name} scenario")

    # Calculate totals
    total_input = sum(r.input_tokens_est for r in results)
    total_output = sum(r.output_tokens_est for r in results)
    cloud_input = sum(r.input_tokens_est for r in results if r.layer == "l3")
    cloud_output = sum(r.output_tokens_est for r in results if r.layer == "l3")
    total = len(results)
    deflected = layer_counts["l1a"] + layer_counts["l1b"] + layer_counts["l2"]
    deflection_rate = (deflected / total * 100) if total > 0 else 0.0
    errors = sum(1 for r in results if r.status != 200)

    # Run validation
    val_passed, val_total, val_failed = run_validation(workspace, args)

    return ScenarioResult(
        scenario=name,
        workspace=str(workspace),
        total_prompts=total,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cloud_input_tokens=cloud_input,
        total_cloud_output_tokens=cloud_output,
        deflection_rate=round(deflection_rate, 1),
        l1a_hits=layer_counts["l1a"],
        l1b_hits=layer_counts["l1b"],
        l2_hits=layer_counts["l2"],
        l3_hits=layer_counts["l3"],
        errors=errors,
        validation_passed=val_passed,
        validation_total=val_total,
        validation_failed=val_failed,
        prompts=results,
    )


def run_validation(workspace: Path, args: argparse.Namespace) -> tuple[bool, int, int]:
    """Run validate_todo_app.py against the workspace. Returns (passed, total, failed)."""
    validator = Path(__file__).parent / "validate_todo_app.py"
    if not validator.exists():
        print("  [warn] validate_todo_app.py not found — skipping validation", file=sys.stderr)
        return True, 0, 0

    json_out = workspace / ".validation.json"
    cmd = [
        sys.executable, str(validator),
        "--app-dir", str(workspace),
        "--json-output", str(json_out),
        "--warn-only",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if json_out.exists():
        try:
            data = json.loads(json_out.read_text())
            return (
                data.get("overall_passed", False),
                data.get("total", 0),
                data.get("required_failed", 0),
            )
        except json.JSONDecodeError:
            pass

    return result.returncode == 0, 0, 0


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def write_tokens_json(
    baseline: ScenarioResult | None,
    isartor: ScenarioResult | None,
    output_path: Path,
) -> None:
    """Write tokens.json with per-prompt and aggregate token accounting."""
    data: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for label, result in [("baseline", baseline), ("isartor", isartor)]:
        if result is None:
            continue
        data[label] = {
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "total_tokens": result.total_input_tokens + result.total_output_tokens,
            "cloud_input_tokens": result.total_cloud_input_tokens,
            "cloud_output_tokens": result.total_cloud_output_tokens,
            "cloud_tokens": result.total_cloud_input_tokens + result.total_cloud_output_tokens,
            "deflection_rate_pct": result.deflection_rate,
            "layer_breakdown": {
                "l1a": result.l1a_hits,
                "l1b": result.l1b_hits,
                "l2": result.l2_hits,
                "l3": result.l3_hits,
            },
            "prompts": [
                {
                    "step": p.step,
                    "phase": p.phase,
                    "layer": p.layer,
                    "input_tokens": p.input_tokens_est,
                    "output_tokens": p.output_tokens_est,
                    "latency_ms": round(p.latency_ms, 1),
                }
                for p in result.prompts
            ],
        }

    if baseline and isartor:
        bl_cloud = baseline.total_cloud_input_tokens + baseline.total_cloud_output_tokens
        is_cloud = isartor.total_cloud_input_tokens + isartor.total_cloud_output_tokens
        saved = bl_cloud - is_cloud
        pct = (saved / bl_cloud * 100) if bl_cloud > 0 else 0
        data["comparison"] = {
            "baseline_cloud_tokens": bl_cloud,
            "isartor_cloud_tokens": is_cloud,
            "tokens_saved": saved,
            "savings_pct": round(pct, 1),
        }

    output_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  Written: {output_path}", file=sys.stderr)


def write_code_diff(ws_baseline: Path, ws_isartor: Path, output_path: Path) -> None:
    """Generate a unified diff between the two workspaces."""
    result = subprocess.run(
        ["diff", "-ruN", "--exclude=.git", "--exclude=.validation.json",
         str(ws_baseline), str(ws_isartor)],
        capture_output=True, text=True,
    )
    output_path.write_text(result.stdout or "(no differences)\n")
    print(f"  Written: {output_path}", file=sys.stderr)


def write_summary_report(
    baseline: ScenarioResult | None,
    isartor: ScenarioResult | None,
    output_path: Path,
) -> None:
    """Write a markdown summary report."""
    lines = ["# Scenario Runner — Code Generation Benchmark\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n")

    for label, result in [("Baseline (Copilot only)", baseline),
                          ("Isartor + Copilot", isartor)]:
        if result is None:
            continue
        lines.append(f"\n## {label}\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Prompts | {result.total_prompts} |")
        lines.append(f"| Total tokens (est.) | {result.total_input_tokens + result.total_output_tokens:,} |")
        lines.append(f"| Cloud tokens | {result.total_cloud_input_tokens + result.total_cloud_output_tokens:,} |")
        lines.append(f"| Deflection rate | {result.deflection_rate}% |")
        lines.append(f"| L1a / L1b / L2 / L3 | {result.l1a_hits} / {result.l1b_hits} / {result.l2_hits} / {result.l3_hits} |")
        lines.append(f"| Errors | {result.errors} |")
        lines.append(f"| Validation | {'✅ passed' if result.validation_passed else '❌ failed'} ({result.validation_total} checks, {result.validation_failed} failed) |")

    if baseline and isartor:
        bl_cloud = baseline.total_cloud_input_tokens + baseline.total_cloud_output_tokens
        is_cloud = isartor.total_cloud_input_tokens + isartor.total_cloud_output_tokens
        saved = bl_cloud - is_cloud
        pct = (saved / bl_cloud * 100) if bl_cloud > 0 else 0
        lines.append("\n## Comparison\n")
        lines.append(f"| Metric | Baseline | Isartor | Savings |")
        lines.append(f"|--------|----------|---------|---------|")
        lines.append(f"| Cloud tokens | {bl_cloud:,} | {is_cloud:,} | {saved:,} ({pct:.0f}%) |")
        lines.append(f"| Cloud requests | {baseline.l3_hits} | {isartor.l3_hits} | {baseline.l3_hits - isartor.l3_hits} |")

    lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"  Written: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Code-generation scenario runner for Isartor benchmarks.",
    )
    p.add_argument("--fixture", default="benchmarks/fixtures/claude_code_todo_app.jsonl",
                   help="Path to the JSONL fixture file.")
    p.add_argument("--output-dir", default="benchmarks/results/scenario_run",
                   help="Directory for workspaces and outputs.")
    p.add_argument("--requests", type=int, default=0,
                   help="Limit number of prompts (0 = all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Simulate responses without calling any API.")

    # Scenario selection
    p.add_argument("--both", action="store_true", default=True,
                   help="Run both baseline and isartor scenarios (default).")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run only the baseline scenario.")
    p.add_argument("--isartor-only", action="store_true",
                   help="Run only the isartor scenario.")

    # Baseline config
    p.add_argument("--direct-url",
                   default=os.environ.get("DIRECT_LLM_URL", ""),
                   help="Direct LLM URL for baseline.")
    p.add_argument("--direct-api-key",
                   default=os.environ.get("DIRECT_LLM_API_KEY",
                           os.environ.get("ANTHROPIC_API_KEY", "")),
                   help="API key for baseline direct requests.")
    p.add_argument("--copilot-token",
                   default=os.environ.get("COPILOT_KEY",
                           os.environ.get("ISARTOR_COPILOT_TOKEN", "")),
                   help="GitHub token for Copilot baseline.")

    # Isartor config
    p.add_argument("--isartor-url",
                   default=os.environ.get("ISARTOR_URL", "http://localhost:8080"),
                   help="Isartor gateway URL.")
    p.add_argument("--api-key",
                   default=os.environ.get("ISARTOR_API_KEY", ""),
                   help="Isartor gateway API key.")

    p.add_argument("--timeout", type=float,
                   default=float(os.environ.get("ISARTOR_TIMEOUT", "120")),
                   help="Request timeout in seconds.")
    p.add_argument("--model", default=os.environ.get("COPILOT_MODEL", "gpt-5.4"),
                   help="Model name for Copilot requests (default: gpt-5.4).")
    p.add_argument("--push-repo",
                   default=os.environ.get("PUSH_REPO", ""),
                   help="Git repo URL to push workspaces (e.g. https://github.com/org/repo).")

    return p


def main() -> int:
    args = build_parser().parse_args()

    # Load fixture
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"Error: fixture not found: {fixture_path}", file=sys.stderr)
        return 1

    entries = []
    with open(fixture_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if args.requests > 0:
        entries = entries[: args.requests]

    print(f"Loaded {len(entries)} prompts from {fixture_path}", file=sys.stderr)

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_baseline = not args.isartor_only
    run_isartor = not args.baseline_only

    # Validate auth before starting
    if not args.dry_run:
        if run_baseline and not args.copilot_token and not args.direct_api_key:
            print("Error: baseline requires --copilot-token or --direct-api-key", file=sys.stderr)
            return 1
        if run_isartor and not args.api_key:
            print("Error: isartor scenario requires --api-key", file=sys.stderr)
            return 1

    baseline_result: ScenarioResult | None = None
    isartor_result: ScenarioResult | None = None

    # --- Baseline scenario ---
    if run_baseline:
        print("\n═══ Scenario A: Baseline (Copilot only) ═══", file=sys.stderr)
        ws_baseline = init_workspace(output_dir, "workspace_baseline")
        baseline_result = run_scenario("baseline", entries, ws_baseline, args)
        print(f"  → {baseline_result.total_prompts} prompts, "
              f"{baseline_result.errors} errors, "
              f"deflection {baseline_result.deflection_rate}%", file=sys.stderr)

    # --- Isartor scenario ---
    if run_isartor:
        print("\n═══ Scenario B: Isartor + Copilot ═══", file=sys.stderr)
        ws_isartor = init_workspace(output_dir, "workspace_isartor")
        isartor_result = run_scenario("isartor", entries, ws_isartor, args)
        print(f"  → {isartor_result.total_prompts} prompts, "
              f"{isartor_result.errors} errors, "
              f"deflection {isartor_result.deflection_rate}%", file=sys.stderr)

    # --- Generate outputs ---
    print("\n═══ Generating outputs ═══", file=sys.stderr)

    write_tokens_json(baseline_result, isartor_result, output_dir / "tokens.json")

    if baseline_result and isartor_result:
        write_code_diff(
            Path(baseline_result.workspace),
            Path(isartor_result.workspace),
            output_dir / "code.diff",
        )

    write_summary_report(baseline_result, isartor_result, output_dir / "summary.md")

    # Write full JSON results
    full_results: dict[str, Any] = {}
    for label, res in [("baseline", baseline_result), ("isartor", isartor_result)]:
        if res:
            d = asdict(res)
            full_results[label] = d
    (output_dir / "full_results.json").write_text(json.dumps(full_results, indent=2) + "\n")

    # Push workspaces to remote repo
    if args.push_repo:
        print("\n═══ Pushing workspaces ═══", file=sys.stderr)
        timestamp = time.strftime("%Y%m%d-%H%M", time.gmtime())
        if baseline_result:
            push_workspace(
                Path(baseline_result.workspace),
                args.push_repo,
                f"benchmark/baseline-{timestamp}",
            )
        if isartor_result:
            push_workspace(
                Path(isartor_result.workspace),
                args.push_repo,
                f"benchmark/isartor-{timestamp}",
            )

    print(f"\n✅ All outputs in {output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

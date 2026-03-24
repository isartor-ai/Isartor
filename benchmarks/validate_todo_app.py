#!/usr/bin/env python3
"""
TypeScript Todo-App Acceptance Validator
=========================================

Validates that a generated TypeScript todo app meets the acceptance criteria
defined in the Claude Code benchmark task packet.

Checks performed (in order):

  1. Required files are present.
  2. package.json is valid JSON and contains expected scripts and dependencies.
  3. tsconfig.json is valid JSON with strict mode enabled.
  4. TypeScript source files satisfy basic structural patterns.
  5. Test files cover the required test cases (string patterns).
  6. Docker artefacts (Dockerfile, docker-compose.yml) are present.
  7. [optional] TypeScript compiles without errors (requires tsc on PATH).
  8. [optional] Jest test suite passes (requires npm test in the app dir).
  9. [optional] API smoke test against a running app (requires --api-url).

Usage
-----
  # Validate a generated app at ./output/todo-app (dry checks only):
  python3 benchmarks/validate_todo_app.py --app-dir ./output/todo-app

  # Full validation including compile + tests:
  python3 benchmarks/validate_todo_app.py --app-dir ./output/todo-app \\
      --compile --run-tests

  # With API smoke test against a running instance:
  python3 benchmarks/validate_todo_app.py --app-dir ./output/todo-app \\
      --api-url http://localhost:3000

  # Machine-readable JSON output:
  python3 benchmarks/validate_todo_app.py --app-dir ./output/todo-app \\
      --json-output ./validation_results.json

Exit codes
----------
  0  All checks passed (or only warnings when --warn-only is used).
  1  One or more required checks failed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Check result dataclass ─────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    optional: bool = False
    detail: str = ""


# ── File presence checks ───────────────────────────────────────────────────────

REQUIRED_FILES = [
    "package.json",
    "tsconfig.json",
    "src/types.ts",
    "src/store.ts",
    "src/app.ts",
    "src/server.ts",
    "src/routes/todos.ts",
    "src/middleware/errorHandler.ts",
    "public/index.html",
    "public/app.js",
    "public/styles.css",
    "__tests__/unit/store.test.ts",
    "__tests__/integration/todos.test.ts",
    "jest.config.js",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    "README.md",
]

OPTIONAL_FILES = [
    ".env.example",
    "src/middleware/notFound.ts",
    ".dockerignore",
]


def check_required_files(app_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    for rel_path in REQUIRED_FILES:
        p = app_dir / rel_path
        results.append(
            CheckResult(
                name=f"file:{rel_path}",
                passed=p.exists(),
                message=f"{'Present' if p.exists() else 'MISSING'}: {rel_path}",
            )
        )
    for rel_path in OPTIONAL_FILES:
        p = app_dir / rel_path
        results.append(
            CheckResult(
                name=f"file:{rel_path}",
                passed=p.exists(),
                message=f"{'Present' if p.exists() else 'Missing (optional)'}: {rel_path}",
                optional=True,
            )
        )
    return results


# ── package.json checks ────────────────────────────────────────────────────────

EXPECTED_SCRIPTS = {"build", "start", "dev", "test"}
EXPECTED_DEPS = {"express", "cors", "uuid"}
EXPECTED_DEV_DEPS = {"typescript", "jest", "ts-jest", "supertest"}


def check_package_json(app_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    pkg_path = app_dir / "package.json"
    if not pkg_path.exists():
        return [CheckResult("package.json:parse", False, "package.json not found")]

    try:
        pkg = json.loads(pkg_path.read_text())
    except json.JSONDecodeError as exc:
        return [CheckResult("package.json:parse", False, f"Invalid JSON: {exc}")]

    results.append(CheckResult("package.json:parse", True, "package.json is valid JSON"))

    scripts = set(pkg.get("scripts", {}).keys())
    missing_scripts = EXPECTED_SCRIPTS - scripts
    results.append(
        CheckResult(
            "package.json:scripts",
            len(missing_scripts) == 0,
            (
                "All expected scripts present"
                if not missing_scripts
                else f"Missing scripts: {', '.join(sorted(missing_scripts))}"
            ),
        )
    )

    deps = set(pkg.get("dependencies", {}).keys())
    missing_deps = EXPECTED_DEPS - deps
    results.append(
        CheckResult(
            "package.json:dependencies",
            len(missing_deps) == 0,
            (
                "All expected dependencies present"
                if not missing_deps
                else f"Missing dependencies: {', '.join(sorted(missing_deps))}"
            ),
        )
    )

    dev_deps = set(pkg.get("devDependencies", {}).keys())
    missing_dev = EXPECTED_DEV_DEPS - dev_deps
    results.append(
        CheckResult(
            "package.json:devDependencies",
            len(missing_dev) == 0,
            (
                "All expected devDependencies present"
                if not missing_dev
                else f"Missing devDependencies: {', '.join(sorted(missing_dev))}"
            ),
        )
    )

    return results


# ── tsconfig.json checks ───────────────────────────────────────────────────────

def check_tsconfig(app_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    tsconfig_path = app_dir / "tsconfig.json"
    if not tsconfig_path.exists():
        return [CheckResult("tsconfig:parse", False, "tsconfig.json not found")]

    try:
        tsconfig = json.loads(tsconfig_path.read_text())
    except json.JSONDecodeError as exc:
        return [CheckResult("tsconfig:parse", False, f"Invalid JSON: {exc}")]

    results.append(CheckResult("tsconfig:parse", True, "tsconfig.json is valid JSON"))

    compiler_opts = tsconfig.get("compilerOptions", {})
    strict = compiler_opts.get("strict", False)
    results.append(
        CheckResult(
            "tsconfig:strict",
            bool(strict),
            f"strict mode: {strict}",
        )
    )

    out_dir = compiler_opts.get("outDir", "")
    results.append(
        CheckResult(
            "tsconfig:outDir",
            bool(out_dir),
            f"outDir set: {out_dir!r}" if out_dir else "outDir not set",
        )
    )

    return results


# ── Source structure checks ────────────────────────────────────────────────────

def _file_contains(path: Path, pattern: str, flags: int = 0, _cache: dict[Path, str] = {}) -> bool:  # noqa: B006
    """Return True if the file content matches the regex pattern.

    File contents are cached on first read so that multiple pattern checks
    against the same file avoid repeated disk I/O.
    """
    if path not in _cache:
        try:
            _cache[path] = path.read_text(errors="replace")
        except OSError:
            _cache[path] = ""
    return bool(re.search(pattern, _cache[path], flags))


def check_source_structure(app_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    # src/types.ts — must define Todo interface
    types_path = app_dir / "src/types.ts"
    results.append(
        CheckResult(
            "src/types.ts:Todo",
            _file_contains(types_path, r"\bTodo\b"),
            "src/types.ts defines Todo type",
        )
    )
    # Must have id, title, completed fields
    for field_name in ("id", "title", "completed"):
        results.append(
            CheckResult(
                f"src/types.ts:{field_name}",
                _file_contains(types_path, rf"\b{field_name}\b"),
                f"src/types.ts has '{field_name}' field",
            )
        )

    # src/store.ts — must export CRUD functions
    store_path = app_dir / "src/store.ts"
    for fn in ("getAllTodos", "createTodo", "updateTodo", "deleteTodo", "getTodoById"):
        results.append(
            CheckResult(
                f"src/store.ts:{fn}",
                _file_contains(store_path, rf"\bexport\b.*\b{fn}\b"),
                f"src/store.ts exports {fn}",
            )
        )

    # src/routes/todos.ts — must define HTTP handlers
    router_path = app_dir / "src/routes/todos.ts"
    for method_path in (
        ("get", r"router\.get\("),
        ("post", r"router\.post\("),
        ("put", r"router\.put\(|router\.patch\("),
        ("delete", r"router\.delete\("),
    ):
        results.append(
            CheckResult(
                f"src/routes/todos.ts:{method_path[0]}",
                _file_contains(router_path, method_path[1]),
                f"src/routes/todos.ts has {method_path[0].upper()} handler",
            )
        )

    # src/app.ts — must register the todos router
    app_path = app_dir / "src/app.ts"
    results.append(
        CheckResult(
            "src/app.ts:todos-router",
            _file_contains(app_path, r"todos") and _file_contains(app_path, r"/api"),
            "src/app.ts mounts todos router under /api",
        )
    )
    results.append(
        CheckResult(
            "src/app.ts:health",
            _file_contains(app_path, r"/health"),
            "src/app.ts exposes /health endpoint",
        )
    )
    results.append(
        CheckResult(
            "src/app.ts:cors",
            _file_contains(app_path, r"\bcors\b"),
            "src/app.ts uses CORS middleware",
        )
    )

    # src/middleware/errorHandler.ts — must have 4-arg signature
    err_path = app_dir / "src/middleware/errorHandler.ts"
    results.append(
        CheckResult(
            "src/middleware/errorHandler.ts:signature",
            _file_contains(
                err_path,
                r"err.*req.*res.*next|NextFunction",
                re.IGNORECASE,
            ),
            "errorHandler.ts has 4-parameter error middleware signature",
        )
    )

    return results


# ── Test file checks ───────────────────────────────────────────────────────────

UNIT_TEST_PATTERNS = [
    ("getAllTodos", r"getAllTodos"),
    ("createTodo", r"createTodo"),
    ("updateTodo", r"updateTodo"),
    ("deleteTodo", r"deleteTodo"),
    ("getTodoById", r"getTodoById"),
    ("resetStore", r"resetStore|beforeEach"),
]

INTEGRATION_TEST_PATTERNS = [
    ("GET /api/todos", r"GET.*\/api\/todos|get.*\/api\/todos"),
    ("POST /api/todos", r"POST.*\/api\/todos|post.*\/api\/todos"),
    ("status 201", r"\.status\(201\)|201"),
    ("status 404", r"\.status\(404\)|404"),
    ("DELETE /api/todos/:id", r"DELETE.*todos|delete.*todos"),
    ("/health endpoint", r"\/health"),
]


def check_test_files(app_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    unit_path = app_dir / "__tests__/unit/store.test.ts"
    for label, pattern in UNIT_TEST_PATTERNS:
        results.append(
            CheckResult(
                f"unit-tests:{label}",
                _file_contains(unit_path, pattern),
                f"store.test.ts covers {label}",
            )
        )

    integ_path = app_dir / "__tests__/integration/todos.test.ts"
    for label, pattern in INTEGRATION_TEST_PATTERNS:
        results.append(
            CheckResult(
                f"integration-tests:{label}",
                _file_contains(integ_path, pattern, re.IGNORECASE),
                f"todos.test.ts covers {label}",
            )
        )

    return results


# ── Frontend checks ────────────────────────────────────────────────────────────

def check_frontend(app_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    html_path = app_dir / "public/index.html"
    for label, pattern in (
        ("DOCTYPE", r"<!DOCTYPE html>"),
        ("viewport", r"viewport"),
        ("todo-list", r"todo-list"),
        ("form", r"<form"),
        ("styles.css", r"styles\.css"),
        ("app.js", r"app\.js"),
    ):
        results.append(
            CheckResult(
                f"public/index.html:{label}",
                _file_contains(html_path, pattern, re.IGNORECASE),
                f"index.html has {label}",
            )
        )

    js_path = app_dir / "public/app.js"
    for label, pattern in (
        ("fetchTodos", r"fetchTodos|fetch.*todos"),
        ("fetch /api/todos", r"\/api\/todos"),
        ("addTodo/POST", r"POST|addTodo"),
        ("deleteTodo/DELETE", r"DELETE|deleteTodo"),
        ("toggleTodo/PATCH", r"PATCH|toggle"),
    ):
        results.append(
            CheckResult(
                f"public/app.js:{label}",
                _file_contains(js_path, pattern, re.IGNORECASE),
                f"app.js implements {label}",
            )
        )

    return results


# ── Docker checks ──────────────────────────────────────────────────────────────

def check_docker(app_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    dockerfile_path = app_dir / "Dockerfile"
    for label, pattern in (
        ("node base image", r"FROM node:"),
        ("multi-stage builder", r"AS builder"),
        ("npm ci", r"npm ci"),
        ("EXPOSE", r"EXPOSE"),
        ("CMD", r"CMD"),
    ):
        results.append(
            CheckResult(
                f"Dockerfile:{label}",
                _file_contains(dockerfile_path, pattern),
                f"Dockerfile has {label}",
            )
        )

    compose_path = app_dir / "docker-compose.yml"
    for label, pattern in (
        ("services", r"services:"),
        ("ports", r"ports:"),
        ("build", r"build:"),
    ):
        results.append(
            CheckResult(
                f"docker-compose.yml:{label}",
                _file_contains(compose_path, pattern),
                f"docker-compose.yml has {label}",
            )
        )

    return results


# ── TypeScript compilation check (optional) ────────────────────────────────────

def check_typescript_compile(app_dir: Path) -> CheckResult:
    """Run tsc --noEmit to verify the project compiles."""
    tsc_bin = app_dir / "node_modules/.bin/tsc"
    cmd = [str(tsc_bin) if tsc_bin.exists() else "tsc", "--noEmit"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(app_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0:
            return CheckResult(
                "tsc:compile",
                True,
                "TypeScript compiles without errors",
                optional=True,
            )
        detail = (proc.stdout + proc.stderr).strip()
        return CheckResult(
            "tsc:compile",
            False,
            "TypeScript compilation failed",
            optional=True,
            detail=detail[:2000],
        )
    except FileNotFoundError:
        return CheckResult(
            "tsc:compile",
            False,
            "tsc not found — run npm install first or install typescript globally",
            optional=True,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            "tsc:compile", False, "tsc timed out after 60 s", optional=True
        )


# ── Jest test suite check (optional) ──────────────────────────────────────────

def check_jest(app_dir: Path) -> CheckResult:
    """Run npm test and check for passing output."""
    try:
        proc = subprocess.run(
            ["npm", "test", "--", "--runInBand", "--forceExit"],
            cwd=str(app_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            return CheckResult(
                "jest:tests",
                True,
                "Jest test suite passed",
                optional=True,
            )
        detail = (proc.stdout + proc.stderr).strip()
        return CheckResult(
            "jest:tests",
            False,
            "Jest test suite failed",
            optional=True,
            detail=detail[:3000],
        )
    except FileNotFoundError:
        return CheckResult(
            "jest:tests",
            False,
            "npm not found on PATH",
            optional=True,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            "jest:tests",
            False,
            "npm test timed out after 120 s",
            optional=True,
        )


# ── API smoke test (optional) ─────────────────────────────────────────────────

def _api_request(
    base_url: str,
    method: str,
    path: str,
    body: dict | None = None,
) -> tuple[int, dict | None]:
    """Return (status_code, parsed_body) or (0, None) on network error."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            try:
                parsed: dict | None = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            return resp.status, parsed
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
            parsed = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            parsed = None
        return exc.code, parsed
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  [warn] API request {method} {path} failed: {exc}", file=sys.stderr)
        return 0, None


def check_api_smoke(base_url: str) -> list[CheckResult]:
    """Run API smoke tests against a running todo app."""
    results: list[CheckResult] = []

    # Health check
    status, body = _api_request(base_url, "GET", "/health")
    results.append(
        CheckResult(
            "api:GET /health",
            status == 200 and isinstance(body, dict) and body.get("status") == "ok",
            f"GET /health → {status}",
            optional=True,
            detail=str(body),
        )
    )

    # GET all todos (initially empty)
    status, body = _api_request(base_url, "GET", "/api/todos")
    results.append(
        CheckResult(
            "api:GET /api/todos (empty)",
            status == 200 and isinstance(body, list),
            f"GET /api/todos → {status}, type={'list' if isinstance(body, list) else type(body).__name__}",
            optional=True,
        )
    )

    # POST create todo
    status, created = _api_request(
        base_url, "POST", "/api/todos", {"title": "Buy groceries"}
    )
    todo_id: str | None = None
    if isinstance(created, dict):
        todo_id = created.get("id")
    results.append(
        CheckResult(
            "api:POST /api/todos",
            status == 201 and todo_id is not None,
            f"POST /api/todos → {status}, id={todo_id!r}",
            optional=True,
        )
    )

    if todo_id:
        # GET by id
        status, body = _api_request(base_url, "GET", f"/api/todos/{todo_id}")
        results.append(
            CheckResult(
                "api:GET /api/todos/:id",
                status == 200 and isinstance(body, dict) and body.get("id") == todo_id,
                f"GET /api/todos/{todo_id} → {status}",
                optional=True,
            )
        )

        # PATCH toggle
        status, body = _api_request(
            base_url, "PATCH", f"/api/todos/{todo_id}", {"completed": True}
        )
        results.append(
            CheckResult(
                "api:PATCH /api/todos/:id",
                status == 200 and isinstance(body, dict) and body.get("completed") is True,
                f"PATCH toggle → {status}",
                optional=True,
            )
        )

        # DELETE
        status, body = _api_request(base_url, "DELETE", f"/api/todos/{todo_id}")
        results.append(
            CheckResult(
                "api:DELETE /api/todos/:id",
                status == 204,
                f"DELETE → {status}",
                optional=True,
            )
        )

        # GET deleted → 404
        status, body = _api_request(base_url, "GET", f"/api/todos/{todo_id}")
        results.append(
            CheckResult(
                "api:GET deleted todo → 404",
                status == 404,
                f"GET deleted todo → {status} (expected 404)",
                optional=True,
            )
        )

    # POST with missing title → 400
    status, body = _api_request(base_url, "POST", "/api/todos", {})
    results.append(
        CheckResult(
            "api:POST invalid → 400",
            status == 400,
            f"POST with empty body → {status} (expected 400)",
            optional=True,
        )
    )

    # 404 route
    status, body = _api_request(base_url, "GET", "/nonexistent-route")
    results.append(
        CheckResult(
            "api:unknown route → 404",
            status == 404,
            f"GET unknown route → {status} (expected 404)",
            optional=True,
        )
    )

    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def _icon(passed: bool, optional: bool) -> str:
    if passed:
        return "✅"
    return "⚠️" if optional else "❌"


def print_report(all_results: list[CheckResult], verbose: bool = False) -> None:
    """Print a human-readable validation report."""
    required_failed = [r for r in all_results if not r.passed and not r.optional]
    optional_failed = [r for r in all_results if not r.passed and r.optional]
    passed = [r for r in all_results if r.passed]

    print(f"\n{'─' * 60}")
    print("  Todo-App Acceptance Validation")
    print(f"{'─' * 60}")
    print(f"  Total checks : {len(all_results)}")
    print(f"  Passed       : {len(passed)}")
    print(f"  Required fail: {len(required_failed)}")
    print(f"  Optional fail: {len(optional_failed)}")
    print()

    for result in all_results:
        if verbose or not result.passed:
            icon = _icon(result.passed, result.optional)
            tag = " [optional]" if result.optional else ""
            print(f"  {icon} {result.name}{tag}: {result.message}")
            if result.detail and not result.passed:
                for line in result.detail.splitlines()[:10]:
                    print(f"       {line}")

    if required_failed:
        print(f"\n  ❌ {len(required_failed)} required check(s) failed.")
    else:
        print("\n  ✅ All required checks passed.")

    if optional_failed:
        print(f"  ⚠️  {len(optional_failed)} optional check(s) failed (non-blocking).")


def build_json_output(
    app_dir: Path,
    all_results: list[CheckResult],
) -> dict[str, Any]:
    required_failed = [r for r in all_results if not r.passed and not r.optional]
    return {
        "app_dir": str(app_dir),
        "total": len(all_results),
        "passed": sum(1 for r in all_results if r.passed),
        "required_failed": len(required_failed),
        "optional_failed": sum(
            1 for r in all_results if not r.passed and r.optional
        ),
        "overall_passed": len(required_failed) == 0,
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "optional": r.optional,
                "message": r.message,
                "detail": r.detail or None,
            }
            for r in all_results
        ],
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TypeScript Todo-App Acceptance Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--app-dir",
        required=True,
        dest="app_dir",
        help="Path to the generated TypeScript todo app directory.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run tsc --noEmit to validate TypeScript compilation (requires node_modules).",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        dest="run_tests",
        help="Run npm test to execute the Jest test suite.",
    )
    parser.add_argument(
        "--api-url",
        dest="api_url",
        default="",
        help="Base URL of a running todo app for API smoke tests (e.g. http://localhost:3000).",
    )
    parser.add_argument(
        "--json-output",
        dest="json_output",
        default="",
        help="Write machine-readable JSON results to this path.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print all check results, not just failures.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        dest="warn_only",
        help="Exit 0 even if required checks fail (useful for CI info steps).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app_dir = Path(args.app_dir).resolve()
    if not app_dir.exists():
        print(f"Error: --app-dir does not exist: {app_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating todo app at: {app_dir}")

    all_results: list[CheckResult] = []

    # 1. File presence
    all_results.extend(check_required_files(app_dir))

    # 2. package.json
    all_results.extend(check_package_json(app_dir))

    # 3. tsconfig.json
    all_results.extend(check_tsconfig(app_dir))

    # 4. Source structure
    all_results.extend(check_source_structure(app_dir))

    # 5. Test files
    all_results.extend(check_test_files(app_dir))

    # 6. Frontend
    all_results.extend(check_frontend(app_dir))

    # 7. Docker
    all_results.extend(check_docker(app_dir))

    # 8. Optional: TypeScript compilation
    if args.compile:
        all_results.append(check_typescript_compile(app_dir))

    # 9. Optional: Jest test suite
    if args.run_tests:
        all_results.append(check_jest(app_dir))

    # 10. Optional: API smoke test
    if args.api_url:
        all_results.extend(check_api_smoke(args.api_url))

    # Report
    print_report(all_results, verbose=args.verbose)

    # JSON output
    if args.json_output:
        out = build_json_output(app_dir, all_results)
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nJSON results written to {out_path}")

    # Exit code
    required_failed = [r for r in all_results if not r.passed and not r.optional]
    if required_failed and not args.warn_only:
        sys.exit(1)


if __name__ == "__main__":
    main()

.PHONY: benchmark benchmark-dry-run \
        benchmark-claude-code benchmark-claude-code-dry-run \
        validate-todo-app \
        build test smoke-claude-copilot

# ── Benchmark targets (existing FAQ / diverse-tasks harness) ──────────────────

## Run the full benchmark suite against a live Isartor instance.
## Requires ISARTOR_URL to be set (default: http://localhost:8080).
## Usage: make benchmark
##        ISARTOR_URL=http://localhost:3000 make benchmark
##        ISARTOR_API_KEY=mysecret make benchmark
benchmark:
	python3 benchmarks/run.py --all \
		--url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}"

## Run the benchmark harness in dry-run mode (no server required).
## Useful for smoke-testing the harness and CI validation.
## Usage: make benchmark-dry-run
benchmark-dry-run:
	python3 benchmarks/run.py --all --dry-run

# ── Claude Code todo-app benchmark ───────────────────────────────────────────

## Run the Claude Code TypeScript todo-app benchmark against a live Isartor
## instance (all three scenarios: baseline, cold, warm).
## Optionally post results to a GitHub issue:
##   GH_TOKEN=ghp_xxx GH_REPO=isartor-ai/Isartor GH_ISSUE=42 make benchmark-claude-code
benchmark-claude-code:
	python3 benchmarks/claude_code_benchmark.py --all-scenarios \
		--url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}" \
		$$([ -n "$${GH_TOKEN}" ] && echo "--gh-token $${GH_TOKEN}") \
		$$([ -n "$${GH_REPO}" ] && echo "--repo $${GH_REPO}") \
		$$([ -n "$${GH_ISSUE}" ] && echo "--issue $${GH_ISSUE}")

## Run the Claude Code todo-app benchmark in dry-run mode (no server required).
## Runs all three scenarios with simulated responses — useful for CI smoke tests.
benchmark-claude-code-dry-run:
	python3 benchmarks/claude_code_benchmark.py --all-scenarios --dry-run

## Validate a generated TypeScript todo app (file presence + structural checks).
## Usage: APP_DIR=./output/todo-app make validate-todo-app
##        APP_DIR=./output/todo-app make validate-todo-app VALIDATE_ARGS="--compile --run-tests"
validate-todo-app:
	python3 benchmarks/validate_todo_app.py \
		--app-dir "$${APP_DIR:-./output/todo-app}" \
		$${VALIDATE_ARGS:-}

# ── Build / test shortcuts ────────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test --all-features

smoke-claude-copilot:
	./scripts/claude-copilot-smoke-test.sh

.PHONY: benchmark benchmark-dry-run \
        benchmark-claude-code benchmark-claude-code-dry-run \
        validate-todo-app \
.PHONY: benchmark benchmark-dry-run report report-dry-run build test smoke-claude-copilot
.PHONY: benchmark benchmark-dry-run benchmark-qwen build test smoke-claude-copilot
.PHONY: benchmark benchmark-dry-run build test smoke-claude-copilot \
        benchmark-claude-code benchmark-claude-code-dry-run benchmark-claude-code-report
.PHONY: benchmark benchmark-dry-run \
        benchmark-claude-copilot benchmark-claude-copilot-dry-run \
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
## Run the Claude Code three-way benchmark (baseline / cold / warm) against a
## live Isartor instance with Qwen 2.5 Coder 7B as Layer 2.
## Requires: Isartor running at ISARTOR_URL with Qwen L2 sidecar enabled.
## Usage: make benchmark-claude-code
##        ISARTOR_URL=http://localhost:8080 ISARTOR_API_KEY=changeme make benchmark-claude-code
benchmark-claude-code:
	./scripts/run_claude_code_benchmark.sh \
		--isartor-url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}"

## Run the Claude Code three-way benchmark in dry-run mode (no server required).
## Produces a realistic three-way comparison report using simulated responses.
## Usage: make benchmark-claude-code-dry-run
benchmark-claude-code-dry-run:
	./scripts/run_claude_code_benchmark.sh --dry-run
## Generate the with/without-Isartor ROI report from existing benchmark results.
## Requires a live benchmark to have been run first (make benchmark).
## Writes benchmarks/results/roi_report.json and benchmarks/results/roi_report.md.
## Usage: make report
report:
	python3 benchmarks/report.py

## Generate the ROI report using simulated (dry-run) data — no server required.
## Useful for CI and offline validation.
## Writes benchmarks/results/roi_report.json and benchmarks/results/roi_report.md.
## Usage: make report-dry-run
report-dry-run:
	python3 benchmarks/report.py --dry-run
## Run the Claude Code / Qwen 2.5 Coder benchmark against a live Isartor instance
## wired to the real Qwen 2.5 Coder 7B sidecar.
## Prerequisites: start the stack first →
##   cd docker && docker compose -f docker-compose.qwen-benchmark.yml up --build
## Usage: make benchmark-qwen
##        ISARTOR_URL=http://localhost:8080 ISARTOR_API_KEY=changeme make benchmark-qwen
benchmark-qwen:
	python3 benchmarks/run.py \
		--url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}" \
		--input benchmarks/fixtures/claude_code_tasks.jsonl \
		--timeout 180
## Run the Claude Code + GitHub Copilot three-scenario benchmark
## (baseline / cold cache / warm cache) against a live Isartor instance.
## Requires Isartor running with the Qwen 2.5 Coder 7B sidecar.
## Usage: make benchmark-claude-code
##        ISARTOR_URL=http://localhost:8080 ISARTOR_API_KEY=changeme make benchmark-claude-code
benchmark-claude-code:
	python3 benchmarks/claude_code_benchmark.py \
		--url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}"

## Run the Claude Code benchmark in dry-run mode (no server required).
## Useful for CI validation and smoke-testing the harness.
## Usage: make benchmark-claude-code-dry-run
benchmark-claude-code-dry-run:
	python3 benchmarks/claude_code_benchmark.py --dry-run

## Generate the ROI markdown report from the latest Claude Code benchmark results.
## Reads benchmarks/results/claude_code_latest.json produced by benchmark-claude-code.
## Usage: make benchmark-claude-code-report
benchmark-claude-code-report:
	python3 benchmarks/roi_report.py

## Run the Claude Code + GitHub Copilot comparison benchmark (Case A vs Case B).
## Case A: direct cloud path (no Isartor).
## Case B: via Isartor with Qwen 2.5 Coder 7B as Layer 2.
## Requires ISARTOR_URL, ISARTOR_API_KEY, and optionally ANTHROPIC_API_KEY.
## Usage: make benchmark-claude-copilot
##        ISARTOR_URL=http://localhost:8080 ISARTOR_API_KEY=changeme make benchmark-claude-copilot
benchmark-claude-copilot:
	./scripts/run_claude_code_benchmark.sh --compare \
		--isartor-url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}"

## Run the Claude Code + GitHub Copilot benchmark in dry-run mode (no server needed).
## Produces a realistic comparison report using simulated responses.
## Useful for CI validation and report format verification.
## Usage: make benchmark-claude-copilot-dry-run
benchmark-claude-copilot-dry-run:
	python3 benchmarks/claude_code_benchmark.py --dry-run

# ── Build / test shortcuts ────────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test --all-features

smoke-claude-copilot:
	./scripts/claude-copilot-smoke-test.sh

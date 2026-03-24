.PHONY: benchmark benchmark-dry-run build test smoke-claude-copilot \
        benchmark-claude-code benchmark-claude-code-dry-run benchmark-claude-code-report

# ── Benchmark targets ─────────────────────────────────────────────────────────

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

# ── Build / test shortcuts ────────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test --all-features

smoke-claude-copilot:
	./scripts/claude-copilot-smoke-test.sh

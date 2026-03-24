.PHONY: benchmark benchmark-dry-run \
        benchmark-claude-code benchmark-claude-code-dry-run \
        build test smoke-claude-copilot

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

# ── Build / test shortcuts ────────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test --all-features

smoke-claude-copilot:
	./scripts/claude-copilot-smoke-test.sh

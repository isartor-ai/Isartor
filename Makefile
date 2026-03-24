.PHONY: benchmark benchmark-dry-run benchmark-qwen build test smoke-claude-copilot

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

# ── Build / test shortcuts ────────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test --all-features

smoke-claude-copilot:
	./scripts/claude-copilot-smoke-test.sh

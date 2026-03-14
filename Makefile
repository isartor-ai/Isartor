.PHONY: benchmark benchmark-dry-run build test

# ── Benchmark targets ─────────────────────────────────────────────────────────

## Run the full benchmark suite against a live Isartor instance.
## Requires ISARTOR_URL to be set (default: http://localhost:8080).
## Usage: make benchmark
##        ISARTOR_URL=http://localhost:3000 make benchmark
benchmark:
	python3 benchmarks/run.py --all --url "$${ISARTOR_URL:-http://localhost:8080}"

## Run the benchmark harness in dry-run mode (no server required).
## Useful for smoke-testing the harness and CI validation.
## Usage: make benchmark-dry-run
benchmark-dry-run:
	python3 benchmarks/run.py --all --dry-run

# ── Build / test shortcuts ────────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test --all-features

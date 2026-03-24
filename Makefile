.PHONY: benchmark benchmark-dry-run report report-dry-run build test smoke-claude-copilot

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

# ── Build / test shortcuts ────────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test --all-features

smoke-claude-copilot:
	./scripts/claude-copilot-smoke-test.sh

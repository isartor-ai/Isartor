.PHONY: benchmark benchmark-dry-run report report-dry-run \
        benchmark-vscode-todo-init benchmark-vscode-todo-smoke \
        benchmark-qwen benchmark-qwen-smoke \
        benchmark-claude-code benchmark-claude-code-dry-run benchmark-claude-code-report \
        benchmark-claude-copilot benchmark-claude-copilot-dry-run \
        validate-todo-app scenario-run scenario-run-dry \
        claude-bench claude-bench-dry \
        build test smoke-claude-copilot

# -- Benchmark targets ---------------------------------------------------------

benchmark:
	python3 benchmarks/run.py --all \
		--url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}" \
		--markdown-output benchmarks/results/latest.md

benchmark-dry-run:
	python3 benchmarks/run.py --all --dry-run \
		--markdown-output benchmarks/results/latest.md

report:
	python3 benchmarks/report.py

report-dry-run:
	python3 benchmarks/report.py --dry-run

benchmark-vscode-todo-init:
	python3 benchmarks/vscode_copilot_todo/harness.py init-run \
		--output "$${OUTPUT:-benchmarks/results/vscode-copilot-todo-run}"

benchmark-vscode-todo-smoke:
	python3 benchmarks/vscode_copilot_todo/harness.py validate-spec
	python3 -m unittest discover -s benchmarks/vscode_copilot_todo/tests -p 'test_*.py'

benchmark-qwen:
	python3 benchmarks/run.py --all \
		--url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}" \
		--output "$${BENCHMARK_OUTPUT:-benchmarks/results/qwen25coder7b-sidecar.json}"

benchmark-qwen-smoke:
	./scripts/smoke-benchmark-layer2.sh \
		--gateway-url "$${ISARTOR_URL:-http://localhost:8080}" \
		--sidecar-url "$${ISARTOR_SIDECAR_URL:-http://localhost:8081}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}"

## Run the Claude Code TypeScript todo-app benchmark against a live Isartor
## instance (all three scenarios: baseline, cold, warm).
benchmark-claude-code:
	python3 benchmarks/claude_code_benchmark.py --all-scenarios \
		--url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}" \
		$$([ -n "$${GH_TOKEN}" ] && echo "--gh-token $${GH_TOKEN}") \
		$$([ -n "$${GH_REPO}" ] && echo "--repo $${GH_REPO}") \
		$$([ -n "$${GH_ISSUE}" ] && echo "--issue $${GH_ISSUE}")

benchmark-claude-code-dry-run:
	python3 benchmarks/claude_code_benchmark.py --all-scenarios --dry-run

benchmark-claude-code-report:
	python3 benchmarks/roi_report.py

## Run the Claude Code + GitHub Copilot comparison benchmark.
benchmark-claude-copilot:
	./scripts/run_claude_code_benchmark.sh --compare \
		--isartor-url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}"

benchmark-claude-copilot-dry-run:
	python3 benchmarks/claude_code_benchmark.py --dry-run

## Validate a generated TypeScript todo app.
validate-todo-app:
	python3 benchmarks/validate_todo_app.py \
		--app-dir "$${APP_DIR:-./output/todo-app}" \
		$${VALIDATE_ARGS:-}

## Run the code-generation scenario runner (baseline vs Isartor comparison).
scenario-run:
	python3 benchmarks/scenario_runner.py \
		--isartor-url "$${ISARTOR_URL:-http://localhost:8080}" \
		--api-key "$${ISARTOR_API_KEY:-changeme}" \
		$$([ -n "$${COPILOT_KEY}" ] && echo "--copilot-token $${COPILOT_KEY}")

scenario-run-dry:
	python3 benchmarks/scenario_runner.py --dry-run

claude-bench:
	python3 benchmarks/claude_bench.py \
		--binary "$${ISARTOR_BINARY:-./target/release/isartor}" \
		--copilot-token "$${COPILOT_KEY}" \
		--model "$${MODEL:-gpt-5.4}"

claude-bench-dry:
	python3 benchmarks/claude_bench.py --dry-run

# -- Build / test shortcuts ----------------------------------------------------

build:
	cargo build --release

test:
	cargo test --all-features

smoke-claude-copilot:
	./scripts/claude-copilot-smoke-test.sh

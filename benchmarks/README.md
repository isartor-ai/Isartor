# Isartor Benchmark Suite

This directory contains the reproducible benchmark harness and fixtures for
measuring Isartor's deflection rate and latency characteristics.

Two benchmark tracks are available:

| Track | Script | Use case |
|-------|--------|----------|
| **General harness** | `run.py` | FAQ / mixed-workload deflection rate via `/api/chat` |
| **Claude Code + Copilot** | `claude_code_benchmark.py` | With-vs-without-Isartor comparison for Claude Code agent sessions via `/v1/messages` |

---

## General Benchmark (FAQ / Mixed Workload)

### Quick Start

```bash
# 1. Start the Isartor server (requires a running instance)
#    Configure a known API key used for the X-API-Key header (defaults to "changeme").
docker run -p 8080:8080 \
    -e ISARTOR__GATEWAY_API_KEY=changeme \
    ghcr.io/isartor-ai/isartor:latest
# or (from source):
ISARTOR__GATEWAY_API_KEY=changeme cargo run --release

# 2. Ensure the benchmark harness uses the same API key (for the X-API-Key header)
export ISARTOR_API_KEY=changeme

# 3. Run both built-in fixtures and save results (make shortcut)
make benchmark

# 4. Run without a server (dry-run / smoke-test)
make benchmark-dry-run

# 5. Run a single fixture with a custom request limit
python3 benchmarks/run.py \
    --url http://localhost:8080 \
    --input benchmarks/fixtures/faq_loop.jsonl \
    --requests 500

# 6. Run the Claude Code / Qwen 2.5 Coder 7B Layer 2 benchmark
#    (requires the Qwen sidecar stack: cd docker && docker compose -f docker-compose.qwen-benchmark.yml up --build)
make benchmark-qwen
```

### Requirements

- Python 3.10+ (uses the built-in `urllib` module — no `pip install` needed)
- A running Isartor instance (or use `--dry-run` for offline validation)

### Fixture Files

| File | Prompts | Description |
|------|---------|-------------|
| `fixtures/faq_loop.jsonl` | 1,000 | Simulates a repetitive FAQ / agent-loop workload. Covers returns, shipping, billing, account management, and more — all with semantic rephrasings. Designed to stress L1a (exact cache) and L1b (semantic cache). |
| `fixtures/diverse_tasks.jsonl` | 500 | Genuine variety: code generation, summarisation, Q&A, data extraction, and creative writing. Represents a realistic mixed-workload with lower deflection than the FAQ loop corpus. |
| `fixtures/claude_code_tasks.jsonl` | 388 | Realistic Claude Code / Copilot workload: coding questions, algorithm explanations, Rust/Go/Python/TypeScript patterns, DevOps tasks, and architecture concepts. Designed to stress Layer 2 (SLM) with a Qwen 2.5 Coder 7B sidecar. Run via `make benchmark-qwen`. |
| `fixtures/claude_code_todo.jsonl` | 105 | Realistic Claude Code session prompts for building a React TypeScript todo application. Covers component scaffolding, custom hooks, tests, routing, and tooling. Designed to model the repetitive, cache-friendly patterns of an AI-assisted coding workflow. |
| `fixtures/claude_code_todo_app.jsonl` | 58 | Deterministic TypeScript todo-app coding workload for the Claude Code + Copilot benchmark. Includes unique implementation prompts, semantic variants, and exact repeats. |

Each file is in [JSONL](https://jsonlines.org/) format — one JSON object per line:

```jsonl
{"prompt": "What is your return policy?"}
{"prompt": "How do returns work?"}
```

## CLI Reference

```
usage: run.py [-h] [--url URL] [--api-key KEY] [--input INPUT]
              [--requests REQUESTS] [--all] [--output OUTPUT] [--dry-run]

Isartor Benchmark Harness

options:
  -h, --help           show this help message and exit
  --url URL            Base URL of the running Isartor instance
                       (default: $ISARTOR_URL or http://localhost:8080)
  --api-key KEY        Value for the X-API-Key header; must match the server's
                       gateway_api_key setting
                       (default: $ISARTOR_API_KEY or 'changeme')
  --input INPUT        Path to a JSONL fixture file to benchmark
  --requests REQUESTS  Limit the number of prompts to send (0 = all)
  --all                Run all built-in fixtures and write results
  --output OUTPUT      Path for the results JSON file
                       (default: benchmarks/results/latest.json)
  --dry-run            Simulate responses locally (no server required).
                       Useful for CI validation and smoke-testing.
```

Both `--url` and `--api-key` honour environment variables:

```bash
ISARTOR_URL=http://localhost:3000 \
ISARTOR_API_KEY=mysecret \
python3 benchmarks/run.py --all
```

## Understanding the Output

```
-- faq_loop --
  Total requests : 1000
  L1a (exact)    :   412  (41.2%)
  L1b (semantic) :   189  (18.9%)
  L2  (SLM)      :     0  ( 0.0%)
  L3  (cloud)    :   399  (39.9%)
  Errors         :     0
  Deflection rate: 60.1%
  P50 latency    :  0.3 ms
  P95 latency    :  820.0 ms
  P99 latency    :  950.0 ms
  Cost saved     : $0.1503  ($0.000150/req)

### faq_loop

| Layer              | Hits   | % of Traffic | Avg Latency (p50) |
|--------------------|--------|--------------|-------------------|
| L1a (exact)        |    412 |       41.2%  |            0.3 ms |
| L1b (semantic)     |    189 |       18.9%  |            3.1 ms |
| L2  (SLM)          |      0 |        0.0%  |                 - |
| L3  (cloud)        |    399 |       39.9%  |          820.0 ms |
| **Total deflected**| **601** | **60.1%** | |
| **Cost saved**     |        |              | **$0.000150/req** |

> Overall latency — P50: 0.3 ms | P95: 820.0 ms | P99: 950.0 ms
```

> **Note:** Overall P50 is sub-millisecond because >60% of requests are served
> from cache (L1a/L1b). P95 and P99 reflect cloud-latency once those
> percentiles fall into the L3 (cloud) bucket.
> Cost formula: `601 deflected × 50 tokens × $0.000005/token = $0.1503 total saved`.
> Divided across all 1,000 requests → `$0.1503 ÷ 1000 = $0.000150/req` (the per-request figure shown in the table).

- **L1a (exact)** — request matched an exact (SHA-256) cache entry
- **L1b (semantic)** — request matched a semantically similar cached entry (cosine similarity)
- **L2 (SLM)** — resolved by the local Small Language Model without a cloud call
- **L3 (cloud)** — forwarded to the configured external LLM provider
- **Deflection rate** — percentage of requests resolved by L1a + L1b + L2 (cloud cost avoided)
- **Cost saved** — estimated USD saved per request using the gpt-4o input token price (`$0.000005/token × avg_prompt_tokens × deflected_requests`)

## Response Headers

Every response from Isartor includes two observability headers:

| Header | Values | Description |
|--------|--------|-------------|
| `X-Isartor-Layer` | `l1a`, `l1b`, `l2`, `l3` | Which layer resolved the request |
| `X-Isartor-Deflected` | `true`, `false` | Whether the cloud call was avoided |

## Cost Estimation Methodology

The harness estimates cloud cost saved using the following formula:

```
tokens_saved = avg_prompt_tokens × (L1a_hits + L1b_hits + L2_hits)
cost_saved   = tokens_saved × 0.000005          # gpt-4o input rate, USD/token
cost_per_req = cost_saved / total_requests
```

- `avg_prompt_tokens` defaults to **50** (a conservative estimate for typical FAQ / agent traffic).
- The gpt-4o input rate of `$0.000005/token` is the public OpenAI pricing as of the benchmark baseline.
- Only input tokens are counted; output tokens are not estimated.

## Results File

After `--all` is used, results are written to `benchmarks/results/latest.json`:

```json
{
  "timestamp": "2025-01-15T10:23:00Z",
  "isartor_version": "0.1.0",
  "hardware": "4-core x86_64, 8 GB RAM, no GPU",
  "fixtures": {
    "faq_loop": {
      "total_requests": 1000,
      "deflection_rate": 0.712,
      "l1a_hits": 423,
      "l1b_hits": 214,
      "l2_hits": 75,
      "l3_hits": 288,
      "l1a_rate": 0.423,
      "l1b_rate": 0.214,
      "l2_rate": 0.075,
      "l3_rate": 0.288,
      "error_count": 0,
      "p50_ms": 0.4,
      "p95_ms": 820.0,
      "p99_ms": 950.0,
      "l1a_p50_ms": 0.35,
      "l1b_p50_ms": 3.1,
      "l2_p50_ms": 130.0,
      "l3_p50_ms": 820.0,
      "tokens_saved": 35600,
      "cost_saved_usd": 0.178,
      "cost_per_req_usd": 0.000178
    },
    "diverse_tasks": {
      "total_requests": 500,
      "deflection_rate": 0.38,
      "l1a_hits": 90,
      "l1b_hits": 100,
      "l2_hits": 0,
      "l3_hits": 310,
      "l1a_rate": 0.18,
      "l1b_rate": 0.20,
      "l2_rate": 0.0,
      "l3_rate": 0.62,
      "error_count": 0,
      "p50_ms": 820.0,
      "p95_ms": 1050.0,
      "p99_ms": 1200.0,
      "l1a_p50_ms": 0.35,
      "l1b_p50_ms": 3.2,
      "l2_p50_ms": null,
      "l3_p50_ms": 820.0,
      "tokens_saved": 9500,
      "cost_saved_usd": 0.0475,
      "cost_per_req_usd": 0.000095
    }
  }
}
```

## Reproducing the Reference Numbers

Any engineer can reproduce results in under 10 minutes:

```bash
# Clone and build
git clone https://github.com/isartor-ai/Isartor.git && cd Isartor
cargo build --release

# Start Isartor with default settings (exact + semantic cache enabled).
# The gateway_api_key defaults to "changeme"; the harness will use the
# same value via $ISARTOR_API_KEY so authentication passes automatically.
# For L3 (cloud) requests, configure an Azure OpenAI backend:
ISARTOR__CACHE_MODE=both \
ISARTOR__LLM_PROVIDER=azure \
ISARTOR__EXTERNAL_LLM_URL=https://<resource>.openai.azure.com \
ISARTOR__EXTERNAL_LLM_API_KEY=<your-azure-key> \
ISARTOR__AZURE_DEPLOYMENT_ID=gpt-4o-mini \
ISARTOR__AZURE_API_VERSION=2024-08-01-preview \
./target/release/isartor &
sleep 5  # wait for the server to start

# Run the full benchmark suite (ISARTOR_API_KEY defaults to 'changeme')
make benchmark
# or equivalently:
# ISARTOR_API_KEY=changeme python3 benchmarks/run.py --url http://localhost:8080 --all
```

If you have configured a custom API key, export it before running:

```bash
export ISARTOR__GATEWAY_API_KEY=your-secret-key  # server
export ISARTOR_API_KEY=your-secret-key            # harness
make benchmark
```

Hardware: 4-core CPU, 8 GB RAM, no GPU. Results will vary based on hardware
and whether L2 SLM inference is enabled.

## CI Integration

The `.github/workflows/benchmark.yml` workflow runs automatically on every PR
targeting `main`. It:

1. Builds Isartor from source.
2. Starts the server and waits for it to become healthy.
3. Runs both fixture corpora through the harness.
4. Posts a formatted result table as a PR comment (updated on subsequent pushes).
5. Uploads `benchmarks/results/ci_run.json` as a workflow artifact.

A `validate-harness` job also runs in dry-run mode on every push to confirm
the harness itself is functioning correctly without requiring a live server.

### Required repository secret

The CI workflow routes L3 (cloud) requests through Azure OpenAI. Add the
following secret to your repository (**Settings → Secrets and variables →
Actions → New repository secret**):

| Secret name            | Value                  |
|------------------------|------------------------|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI key  |

Without this secret the server cannot reach the Azure backend and L3 requests
will return 502 errors. L1a/L1b cache-hit rows are unaffected.

## Claude Code + GitHub Copilot Benchmark

A dedicated three-scenario benchmark measures Isartor's impact on a real-world
Claude Code coding session.  It uses the `claude_code_todo.jsonl` fixture, which
simulates a developer using Claude Code to build a React TypeScript todo app.

### Scenarios

| Scenario | Description |
|----------|-------------|
| `baseline` | Requests go directly to L3 (no Isartor) — establishes the no-proxy control. |
| `cold` | First pass through Isartor with an empty cache — measures cold-start overhead. |
| `warm` | Second pass of the same prompts — measures steady-state cache deflection. |

### Quick Start

```bash
# 1. Start Isartor with the Qwen 2.5 Coder 7B sidecar (Layer 2)
cd docker
docker compose \
  -f docker-compose.yml \
  -f docker-compose.qwen-sidecar.yml \
  up -d

# The Qwen model is ~4.4 GB and downloads on first start.
# Wait for the health check to pass before continuing:
docker compose -f docker-compose.qwen-sidecar.yml logs -f qwen-sidecar

# 2. Run all three scenarios
make benchmark-claude-code

# Or without a running server (dry-run / CI validation):
make benchmark-claude-code-dry-run

# 3. Generate the ROI markdown report
make benchmark-claude-code-report
---

## Claude Code + GitHub Copilot Benchmark

`benchmarks/claude_code_benchmark.py` runs a dedicated comparison benchmark that
quantifies the ROI of routing Claude Code through Isartor compared to a direct
cloud path.

### What it measures

- **Case A — without Isartor:** every prompt goes directly to the cloud LLM
  provider (Anthropic API or GitHub Copilot-backed endpoint). All latency is
  cloud-round-trip latency and all tokens consume cloud quota.
- **Case B — with Isartor (Qwen 2.5 Coder 7B as L2):** prompts route through
  the full Isartor deflection stack:
  L1a exact cache → L1b semantic cache → L2 Qwen (llama.cpp) → L3 cloud.

Reported metrics for each case:

| Metric | Description |
|--------|-------------|
| L1a / L1b / L2 / L3 hits | Requests resolved at each layer |
| Deflection rate | % of requests that avoided cloud |
| P50 / P95 / P99 latency | Overall and per-layer latencies |
| Est. cloud tokens avoided | Input + output tokens saved by deflection |
| Est. cost saving | USD reduction using Claude 3.5 Sonnet pricing |

### Quick Start

```bash
# Dry-run — no server needed, CI-safe, deterministic output:
make benchmark-claude-copilot-dry-run

# Case B only against a live Isartor instance:
python3 benchmarks/claude_code_benchmark.py --case B \
    --isartor-url http://localhost:8080

# Full comparison with a real Anthropic API key (Case A) and live Isartor (Case B):
ANTHROPIC_API_KEY=sk-ant-... \
python3 benchmarks/claude_code_benchmark.py --compare \
    --isartor-url http://localhost:8080 \
    --api-key changeme

# Full end-to-end orchestration (downloads model, starts sidecar + Isartor):
GITHUB_TOKEN=ghp_... \
./scripts/run_claude_code_benchmark.sh --compare \
    --start-llama-server \
    --start-isartor
```

### Model setup (Layer 2 — Qwen 2.5 Coder 7B)

```bash
# 1. Download the Qwen 2.5 Coder 7B GGUF (~4.7 GB):
./scripts/download_qwen_gguf.sh

# 2. Start llama-server:
llama-server \
  --model models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8090 \
  --ctx-size 4096 --n-predict 512

# 3. Start Isartor with Qwen as Layer 2:
ISARTOR__ENABLE_SLM_ROUTER=true \
ISARTOR__LAYER2__SIDECAR_URL=http://127.0.0.1:8090/v1 \
./target/release/isartor up
```

### CLI Reference

```
usage: claude_code_benchmark.py [-h] [--url URL] [--api-key KEY]
                                 [--input INPUT] [--requests REQUESTS]
                                 [--scenario {baseline,cold,warm,all}]
                                 [--output OUTPUT] [--timeout TIMEOUT]
                                 [--dry-run]

options:
  --url URL            Base URL of the running Isartor instance
  --api-key KEY        X-API-Key header value
  --input INPUT        Path to a JSONL fixture file
  --requests REQUESTS  Limit prompts per scenario (0 = all)
  --scenario           Which scenario(s) to run (default: all)
  --dry-run            Simulate responses locally — no server required
```

### Acceptance Criteria

The harness enforces these criteria and exits non-zero if any fail:

| Criterion | Threshold |
|-----------|-----------|
| Warm deflection rate | ≥ 60 % |
| Cold deflection rate | ≥ 10 % |
| Error rate (any scenario) | < 5 % |

### Qwen 2.5 Coder 7B Sidecar

`docker/docker-compose.qwen-sidecar.yml` defines the Layer 2 sidecar:

- **Model**: `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` (Q4\_K\_M, ~4.4 GB)
- **Served by**: `ghcr.io/ggml-org/llama.cpp:server`
- **Port**: 8081 (OpenAI-compatible `/v1/chat/completions`)
- **Environment overrides**: `QWEN_CTX_SIZE`, `QWEN_N_GPU_LAYERS`, `QWEN_N_THREADS`, `QWEN_PORT`

Smoke test:

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

### ROI Report

`benchmarks/roi_report.py` reads `benchmarks/results/claude_code_latest.json`
and emits:

- **`benchmarks/results/claude_code_roi_report.md`** — human-readable report
- **`benchmarks/results/claude_code_roi_<timestamp>.json`** — machine-readable artifact

Key assumptions (edit `roi_report.py` to match your environment):

| Assumption | Default |
|------------|---------|
| gpt-4o input price | $0.000005 / token |
| gpt-4o output price | $0.000015 / token |
| Avg input tokens / request | 75 |
| Avg output tokens / response | 300 |
| Monthly request volume | 50,000 |
| Isartor self-hosting cost | $50 / month |
usage: claude_code_benchmark.py [-h] [--case {A,B}] [--compare] [--dry-run]
                                 [--isartor-url URL] [--api-key KEY]
                                 [--direct-url URL] [--direct-api-key KEY]
                                 [--input FILE] [--requests N]
                                 [--output FILE] [--report FILE]

Options:
  --case {A,B}       Run a single case: A (without Isartor) or B (with Isartor)
  --compare          Run both cases and generate a comparison report
  --dry-run          Simulate responses locally — no server needed (CI-safe)
  --isartor-url URL  Isartor base URL (default: $ISARTOR_URL or http://localhost:8080)
  --api-key KEY      Isartor X-API-Key (default: $ISARTOR_API_KEY or 'changeme')
  --direct-url URL   Direct API URL for Case A (default: $ANTHROPIC_BASE_URL)
  --direct-api-key KEY  API key for Case A (default: $ANTHROPIC_API_KEY)
  --input FILE       JSONL fixture file (default: fixtures/claude_code_todo_app.jsonl)
  --requests N       Limit number of prompts (0 = all)
  --output FILE      JSON results file (default: results/claude_code_copilot.json)
  --report FILE      Markdown report file (default: results/claude_code_copilot_report.md)
```

### Output files

| File | Description |
|------|-------------|
| `results/claude_code_copilot.json` | Machine-readable results (both cases) |
| `results/claude_code_copilot_report.md` | Human-readable Markdown comparison report |

### Sample output (dry-run)

```
-- Case A — without Isartor --
  Total requests : 58
  L3  (cloud)    :    58  (100.0%)
  Deflection rate: 0.0%  (no local deflection — every request hits cloud)
  P50 latency    : 1408.5 ms
  Est. cloud cost: $0.3132  ($0.005400/req)

-- Case B — with Isartor (Qwen L2) --
  Total requests : 58
  L1a (exact)    :    20  (34.5%)
  L1b (semantic) :    15  (25.9%)
  L2  (Qwen)     :    11  (19.0%)
  L3  (cloud)    :    12  (20.7%)
  Deflection rate: 79.3%
  P50 latency    : 5.9 ms
  Est. cloud cost: $0.0648  ($0.001117/req)
```

> **Note on Case A control path:** Claude Code's native behavior routes requests
> through GitHub Copilot's infrastructure, which does not expose per-request
> layer or deflection metadata. The Case A baseline therefore uses a direct
> Anthropic API call (or a simulated all-L3 distribution in dry-run mode) as
> the nearest defensible control. This is documented explicitly in the
> benchmark report so comparisons are reproducible and transparent.


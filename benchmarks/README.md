# Isartor Benchmark Suite

This directory contains the reproducible benchmark harness and fixtures for
measuring Isartor's deflection rate and latency characteristics.

Two benchmark tracks are available:

| Track | Script | Use case |
|-------|--------|----------|
| **General harness** | `run.py` | FAQ / mixed-workload deflection rate via `/api/chat` |
| **Claude Code 3-way** | `claude_code_benchmark.py` | Baseline vs cold vs warm comparison for Claude Code agent sessions via `/v1/messages` |

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
```

## Requirements

- Python 3.10+ (uses the built-in `urllib` module — no `pip install` needed)
- A running Isartor instance (or use `--dry-run` for offline validation)

## Fixture Files

| File | Prompts | Description |
|------|---------|-------------|
| `fixtures/faq_loop.jsonl` | 1,000 | Simulates a repetitive FAQ / agent-loop workload. Covers returns, shipping, billing, account management, and more — all with semantic rephrasings. Designed to stress L1a (exact cache) and L1b (semantic cache). |
| `fixtures/diverse_tasks.jsonl` | 500 | Genuine variety: code generation, summarisation, Q&A, data extraction, and creative writing. Represents a realistic mixed-workload with lower deflection than the FAQ loop corpus. |
| `fixtures/claude_code_todo_app.jsonl` | 58 | Deterministic TypeScript todo-app coding workload for the Claude Code three-way benchmark. Includes unique implementation prompts, semantic variants, and exact repeats. |

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

---

## Claude Code Three-Way Benchmark

`benchmarks/claude_code_benchmark.py` runs a three-way comparison benchmark
that quantifies the ROI of routing Claude Code through Isartor.

### What it measures

- **Baseline — without Isartor:** every prompt goes directly to the cloud LLM.
  No local deflection. All latency is cloud-round-trip latency.
- **Isartor cold cache:** first pass through Isartor with an empty cache.
  Novel prompts fall through to L2 (Qwen) or L3 (cloud). Only exact duplicate
  prompts within the run hit L1a.
- **Isartor warm cache:** second pass with the cache populated from the cold
  run. Previously-seen prompts are deflected locally at L1a or L1b.

### Quick Start

```bash
# Dry-run — no server needed, CI-safe, deterministic output:
make benchmark-claude-code-dry-run

# Live three-way benchmark against a running Isartor instance:
ISARTOR_URL=http://localhost:8080 ISARTOR_API_KEY=changeme \
  make benchmark-claude-code

# Full end-to-end with auto-start (requires Qwen GGUF + llama-server):
GITHUB_TOKEN=ghp_... \
  ./scripts/run_claude_code_benchmark.sh \
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

### GitHub Actions workflow

The `.github/workflows/claude-code-benchmark.yml` workflow runs the three-way
benchmark on demand and posts progress + final results as GitHub issue comments.

**Trigger:**
```bash
gh workflow run claude-code-benchmark.yml \
  -f issue_number=<N> \
  -f dry_run=true
```

**Issue comment sequence:**
1. 🚀 Benchmark started
2. ⚙️ Environment setup complete (live mode)
3. ✅ Baseline run completed
4. ✅ Isartor cold cache run completed
5. ✅ Isartor warm cache run completed
6. 📊 Final results table

### Output files

| File | Description |
|------|-------------|
| `results/claude_code_benchmark.json` | Machine-readable three-way results |
| `results/claude_code_benchmark_report.md` | Human-readable Markdown report |

### CLI Reference

```
usage: claude_code_benchmark.py [-h] [--three-way] [--scenario {baseline,cold,warm}]
                                  [--dry-run] [--isartor-url URL] [--api-key KEY]
                                  [--direct-url URL] [--direct-api-key KEY]
                                  [--input FILE] [--requests N]
                                  [--output FILE] [--report FILE]

Options:
  --three-way           Run all three scenarios and generate a comparison report
  --scenario {baseline,cold,warm}
                        Run a single scenario
  --dry-run             Simulate responses locally — no server required (CI-safe)
  --isartor-url URL     Isartor base URL (default: $ISARTOR_URL or http://localhost:8080)
  --api-key KEY         Isartor X-API-Key (default: $ISARTOR_API_KEY or 'changeme')
  --direct-url URL      Direct API URL for baseline (default: $ANTHROPIC_BASE_URL)
  --direct-api-key KEY  API key for baseline (default: $ANTHROPIC_API_KEY)
  --input FILE          JSONL fixture file (default: fixtures/claude_code_todo_app.jsonl)
  --requests N          Limit number of prompts per scenario (0 = all)
  --output FILE         JSON results file (default: results/claude_code_benchmark.json)
  --report FILE         Markdown report file (default: results/claude_code_benchmark_report.md)
```

### Sample output (dry-run)

```
-- Baseline — without Isartor --
  Total requests : 58
  L3  (cloud)    :    58  (100.0%)
  Deflection rate: 0.0%  (no local deflection — every request hits cloud)
  P50 latency    : 1421.0 ms
  Est. cloud cost: $0.3132  ($0.005400/req)

-- Isartor cold cache — with Qwen L2 --
  Total requests : 58
  L1a (exact)    :     6  (10.3%)
  L1b (semantic) :     3  ( 5.2%)
  L2  (Qwen)     :     6  (10.3%)
  L3  (cloud)    :    43  (74.1%)
  Deflection rate: 25.9%
  P50 latency    : 1454.8 ms
  Est. cloud cost: $0.2322  ($0.004003/req)

-- Isartor warm cache — with Qwen L2 --
  Total requests : 58
  L1a (exact)    :    21  (36.2%)
  L1b (semantic) :     9  (15.5%)
  L2  (Qwen)     :     5  ( 8.6%)
  L3  (cloud)    :    23  (39.7%)
  Deflection rate: 60.3%
  P50 latency    :  7.4 ms
  Est. cloud cost: $0.1242  ($0.002141/req)
```

# Isartor Benchmark Suite

This directory contains the reproducible benchmark harness and fixtures for
measuring Isartor's deflection rate and latency characteristics.

## Quick Start

```bash
# 1. Start the Isartor server (requires a running instance)
#    Configure a known API key used for the X-API-Key header (defaults to "changeme").
docker run -p 8080:8080 \
    -e ISARTOR__GATEWAY_API_KEY=changeme \
    ghcr.io/isartor-ai/isartor:latest
# or (from source):
ISARTOR__GATEWAY_API_KEY=changeme cargo run --release

# 2. Ensure the benchmark harness uses the same API key (for the X-API-Key header)
export ISARTOR__GATEWAY_API_KEY=changeme

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
| `fixtures/diverse_tasks.jsonl` | 500 | Genuine variety: code generation, summarisation, Q&A, data extraction, and creative writing. Represents realistic lower-deflection traffic. |

Each file is in [JSONL](https://jsonlines.org/) format — one JSON object per line:

```jsonl
{"prompt": "What is your return policy?"}
{"prompt": "How do returns work?"}
```

## CLI Reference

```
usage: run.py [-h] [--url URL] [--input INPUT] [--requests REQUESTS] [--all]
              [--output OUTPUT] [--dry-run]

Isartor Benchmark Harness

options:
  -h, --help           show this help message and exit
  --url URL            Base URL of the running Isartor instance
                       (default: $ISARTOR_URL or http://localhost:8080)
  --input INPUT        Path to a JSONL fixture file to benchmark
  --requests REQUESTS  Limit the number of prompts to send (0 = all)
  --all                Run all built-in fixtures and write results
  --output OUTPUT      Path for the results JSON file
                       (default: benchmarks/results/latest.json)
  --dry-run            Simulate responses locally (no server required).
                       Useful for CI validation and smoke-testing.
```

The `--url` flag also honours the `ISARTOR_URL` environment variable:

```bash
ISARTOR_URL=http://localhost:3000 python3 benchmarks/run.py --all
```

## Understanding the Output

```
-- faq_loop --
  Total requests : 1000
  L1a (exact)    :   412  (41.2%)
  L1b (semantic) :  189   (18.9%)
  L2  (SLM)      :    0   ( 0.0%)
  L3  (cloud)    :   399  (39.9%)
  Errors         :     0
  Deflection rate: 60.1%
  P50 latency    :  0.3 ms
  P95 latency    :  8.5 ms
  P99 latency    :  9.1 ms
  Cost saved     : $0.1503  ($0.000150/req)

### faq_loop

| Layer              | Hits   | % of Traffic | Avg Latency (p50) |
|--------------------|--------|--------------|-------------------|
| L1a (exact)        |    412 |       41.2%  |            0.3 ms |
| L1b (semantic)     |    189 |       18.9%  |            3.1 ms |
| L2  (SLM)          |      0 |        0.0%  |               -   |
| L3  (cloud)        |    399 |       39.9%  |            8.2 ms |
| **Total deflected**| **601** | **60.1%** | |
| **Cost saved**     |        |              | **$0.000150/req** |

> Overall latency — P50: 0.3 ms | P95: 8.5 ms | P99: 9.1 ms
```

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
  "hardware": "4-core CPU, 8GB RAM, no GPU",
  "fixtures": {
    "faq_loop": {
      "total_requests": 1000,
      "deflection_rate": 0.71,
      "l1a_hits": 412,
      "l1b_hits": 189,
      "l2_hits": 109,
      "l3_hits": 290,
      "l1a_rate": 0.412,
      "l1b_rate": 0.189,
      "l2_rate": 0.109,
      "l3_rate": 0.290,
      "error_count": 0,
      "p50_ms": 1.2,
      "p95_ms": 4.8,
      "p99_ms": 9.1,
      "tokens_saved": 35500,
      "cost_saved_usd": 0.1775,
      "cost_per_req_usd": 0.0001775
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
      "p50_ms": 2.1,
      "p95_ms": 6.3,
      "p99_ms": 11.4,
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

# Start Isartor with default settings (exact + semantic cache enabled)
# and a known gateway API key (used for the X-API-Key header).
ISARTOR__CACHE_MODE=both ISARTOR__GATEWAY_API_KEY=changeme ./target/release/isartor &
sleep 5  # wait for the server to start

# Ensure the benchmark harness uses the same API key
export ISARTOR__GATEWAY_API_KEY=changeme

# Run the full benchmark suite
make benchmark
# or equivalently:
# python3 benchmarks/run.py --url http://localhost:8080 --all
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

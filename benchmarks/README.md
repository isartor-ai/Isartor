# Isartor Benchmark Suite

This directory contains the reproducible benchmark harness and fixtures for
measuring Isartor's deflection rate and latency characteristics.

## Quick Start

```bash
# 1. Start the Isartor server (requires a running instance)
docker run -p 8080:8080 ghcr.io/isartor-ai/isartor:latest
# or: cargo run --release

# 2. Run both built-in fixtures and save results
python benchmarks/run.py --url http://localhost:8080 --all

# 3. Run a single fixture with a custom request limit
python benchmarks/run.py \
    --url http://localhost:8080 \
    --input benchmarks/fixtures/faq_loop.jsonl \
    --requests 500
```

## Requirements

- Python 3.10+ (uses the built-in `urllib` module — no `pip install` needed)
- A running Isartor instance

## Fixture Files

| File | Prompts | Description |
|------|---------|-------------|
| `fixtures/faq_loop.jsonl` | 500 | Simulates a repetitive FAQ / agent-loop workload. 20 unique questions × 25 rephrasings. Designed to stress L1a (exact cache) and L1b (semantic cache). |
| `fixtures/diverse_tasks.jsonl` | 500 | Genuine variety: code generation, summarisation, Q&A, data extraction, and creative writing. Represents realistic lower-deflection traffic. |

Each file is in [JSONL](https://jsonlines.org/) format — one JSON object per line:

```jsonl
{"prompt": "What is your return policy?"}
{"prompt": "How do returns work?"}
```

## CLI Reference

```
usage: run.py [-h] [--url URL] [--input INPUT] [--requests REQUESTS] [--all]
              [--output OUTPUT]

Isartor Benchmark Harness

options:
  -h, --help           show this help message and exit
  --url URL            Base URL of the running Isartor instance
                       (default: http://localhost:8080)
  --input INPUT        Path to a JSONL fixture file to benchmark
  --requests REQUESTS  Limit the number of prompts to send (0 = all)
  --all                Run all built-in fixtures and write results
  --output OUTPUT      Path for the results JSON file
                       (default: benchmarks/results/latest.json)
```

## Understanding the Output

```
── faq_loop ──
  Total requests : 500
  L1a (exact)    :  210  (42.0%)
  L1b (semantic) :  145  (29.0%)
  L2  (SLM)      :    0  ( 0.0%)
  L3  (cloud)    :  145  (29.0%)
  Errors         :    0
  Deflection rate: 71.0%
  P50 latency    :  1.2 ms
  P95 latency    :  4.8 ms
```

- **L1a (exact)** — request matched an exact (SHA-256) cache entry
- **L1b (semantic)** — request matched a semantically similar cached entry (cosine similarity)
- **L2 (SLM)** — resolved by the local Small Language Model without a cloud call
- **L3 (cloud)** — forwarded to the configured external LLM provider
- **Deflection rate** — percentage of requests resolved by L1a + L1b + L2 (cloud cost avoided)

## Response Headers

Every response from Isartor includes two observability headers:

| Header | Values | Description |
|--------|--------|-------------|
| `X-Isartor-Layer` | `l1a`, `l1b`, `l2`, `l3` | Which layer resolved the request |
| `X-Isartor-Deflected` | `true`, `false` | Whether the cloud call was avoided |

## Results File

After `--all` is used, results are written to `benchmarks/results/latest.json`:

```json
{
  "timestamp": "2025-01-15T10:23:00Z",
  "isartor_version": "0.1.0",
  "hardware": "4-core CPU, 8GB RAM, no GPU",
  "fixtures": {
    "faq_loop": {
      "total_requests": 500,
      "deflection_rate": 0.71,
      "l1a_rate": 0.42,
      "l1b_rate": 0.29,
      "p50_ms": 1.2,
      "p95_ms": 4.8
    },
    "diverse_tasks": {
      "total_requests": 500,
      "deflection_rate": 0.38,
      "l1a_rate": 0.18,
      "l1b_rate": 0.20,
      "p50_ms": 2.1,
      "p95_ms": 6.3
    }
  }
}
```

## Reproducing the Reference Numbers

The reference numbers in `results/latest.json` were produced with:

```bash
# Start Isartor with default settings (exact + semantic cache enabled)
ISARTOR__CACHE_MODE=both cargo run --release &
sleep 5  # wait for the server to start

# Run the benchmark suite
python benchmarks/run.py --url http://localhost:8080 --all
```

Hardware: 4-core CPU, 8 GB RAM, no GPU. Results will vary based on hardware
and whether L2 SLM inference is enabled.

# Testing

> **Complete test runbook for Isartor — from automated test suites to manual feature verification and Copilot CLI integration testing.**

---

## Prerequisites

| Requirement | Check |
|---|---|
| Rust toolchain | `cargo --version` |
| Built binary | `cargo build --release` |
| `curl` + `jq` | `curl --version && jq --version` |

---

## Quick Start — Automated

### Unit & Integration Tests

```bash
# Run the full test suite
cargo test --all-features

# Run a specific test binary
cargo test --test unit_suite
cargo test --test integration_suite
cargo test --test scenario_suite

# Run a single test with output
cargo test --test scenario_suite deflection_rate_at_least_60_percent -- --nocapture
cargo test --test integration_suite body_survives_all_middleware -- --nocapture
```

### Smoke Test Script

Run the entire manual test suite in one command:

```bash
# Start a fresh server, run all tests, stop after
./scripts/smoke-test.sh --stop-after

# Test an already-running server
./scripts/smoke-test.sh --no-start

# Full run including demo + verbose response bodies
./scripts/smoke-test.sh --run-demo --verbose

# Custom URL / API key
./scripts/smoke-test.sh --url http://localhost:9090 --api-key mykey --no-start
```

### Lint & Format Checks

Run the same checks CI runs:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

---

## Manual Step-by-Step

> **Note:** Isartor runs without gateway auth by default (local-first). The test commands below explicitly set `ISARTOR__GATEWAY_API_KEY` to exercise authenticated request handling.

### 1  Start the Server

```bash
# Gateway-only startup (local API testing)
ISARTOR__FIRST_RUN_COMPLETE=1 \
./target/release/isartor up

# Full startup for proxy-aware testing (recommended for this guide)
ISARTOR__FIRST_RUN_COMPLETE=1 \
ISARTOR__GATEWAY_API_KEY=changeme \
./target/release/isartor up copilot

# With an OpenAI key (enables real L3 fallback)
ISARTOR__FIRST_RUN_COMPLETE=1 \
ISARTOR__GATEWAY_API_KEY=changeme \
ISARTOR__EXTERNAL_LLM_API_KEY=sk-... \
./target/release/isartor up copilot
```

Server is ready when you see:
```
INFO isartor: API gateway listening, addr: 0.0.0.0:8080
INFO isartor: CONNECT proxy starting, addr: 0.0.0.0:8081
```

---

### 2  Health & Liveness

```bash
# Liveness probe (no auth needed)
curl http://localhost:8080/healthz

# Rich health (shows layer status, proxy, prompt totals)
curl http://localhost:8080/health | jq .
```

Expected `/health` response shape:
```json
{
  "status": "ok",
  "version": "0.1.25",
  "layers": { "l1a": "active", "l1b": "active", "l2": "active", "l3": "no_api_key" },
  "uptime_seconds": 5,
  "proxy": "active",
  "proxy_layer3": "native_upstream_passthrough",
  "prompt_total_requests": 0,
  "prompt_total_deflected_requests": 0
}
```

---

### 3  OpenAI-Compatible Endpoint (`/v1/chat/completions`)

```bash
API_KEY=changeme

curl -sS http://localhost:8080/v1/chat/completions \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }' | jq .
```

Send the **same prompt twice** to confirm L1a exact-cache kicks in:

```bash
for i in 1 2; do
  echo "--- Request $i ---"
  curl -sS http://localhost:8080/v1/chat/completions \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is 2+2?"}]}' \
    | jq '.choices[0].message.content, .model'
done
```

---

### 4  Anthropic-Compatible Endpoint (`/v1/messages`)

```bash
curl -sS http://localhost:8080/v1/messages \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-haiku-20240307",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }' | jq .
```

Expected shape: `{"id":..., "type":"message", "role":"assistant", "content":[...], "model":...}`

---

### 5  Native Endpoint (`/api/chat`)

```bash
curl -sS http://localhost:8080/api/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "ping"}]}' | jq .
```

---

### 6  L1a — Exact Cache Hit

```bash
# Seed the cache with first request
curl -sS http://localhost:8080/v1/chat/completions \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"capital of France?"}]}' \
  -o /dev/null

# Second identical request — should be served from L1a
curl -sS http://localhost:8080/v1/chat/completions \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"capital of France?"}]}' \
  | jq '.model'
# → "isartor-cache" or similar (not "gpt-4o-mini")
```

---

### 7  L1b — Semantic Cache Hit

```bash
# Seed
curl -sS http://localhost:8080/v1/chat/completions \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is the capital of France?"}]}' \
  -o /dev/null

# Paraphrase — should hit L1b (cosine similarity ≥ 0.85)
curl -sS http://localhost:8080/v1/chat/completions \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Which city is France capital?"}]}' \
  | jq '.model'
```

---

### 8  Authentication Rejection

```bash
# No API key — should return 401/403
curl -sS -w "\nHTTP %{http_code}" http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

---

### 9  Prompt Stats

```bash
# JSON endpoint
curl -sS -H "X-API-Key: $API_KEY" \
  "http://localhost:8080/debug/stats/prompts?limit=10" | jq .

# Per-agent observability endpoint
curl -sS -H "X-API-Key: $API_KEY" \
  "http://localhost:8080/debug/stats/agents" | jq .

# CLI command
./target/release/isartor stats \
  --gateway-url http://localhost:8080 \
  --gateway-api-key $API_KEY

# CLI per-agent view
./target/release/isartor stats \
  --gateway-url http://localhost:8080 \
  --gateway-api-key $API_KEY \
  --by-tool
```

Expected `isartor stats` output:
```
Isartor Prompt Stats
  URL:        http://localhost:8080
  Total:      7
  Deflected:  3

By Layer
  L1A  3
  L3   4

By Surface
  gateway  7

By Client
  openai   5
  anthropic 2

Recent Prompts
  2026-03-19T09:00:00Z gateway openai L1A via /v1/chat/completions (1ms, HTTP 200)
```

---

### 10  Proxy Recent Decisions

```bash
curl -sS -H "X-API-Key: $API_KEY" \
  "http://localhost:8080/debug/proxy/recent?limit=5" | jq .
```

---

### 11  isartor connect status

```bash
./target/release/isartor connect status \
  --gateway-url http://localhost:8080 \
  --gateway-api-key $API_KEY
```

---

### 12  Run the Built-in Demo

```bash
./target/release/isartor demo
# Replays 50 bundled prompts through L1a/L1b, prints deflection rate.
# Writes isartor_demo_result.txt
```

---

### 13  Stop the Server

```bash
./target/release/isartor stop
```

---

## Copilot CLI Integration Test

### Step 1 — Connect Copilot CLI

```bash
./target/release/isartor connect copilot \
  --gateway-url http://localhost:8080 \
  --gateway-api-key changeme
```

This writes `~/.isartor/env/copilot.sh` with:
```bash
export HTTPS_PROXY="http://localhost:8081"
export NODE_EXTRA_CA_CERTS="/Users/<you>/.isartor/ca/isartor-ca.pem"
export ISARTOR_COPILOT_ENABLED=true
```

### Step 2 — Activate the Proxy Environment

**Critical:** You must source the env file **in the same shell** where you run Copilot CLI:

```bash
source ~/.isartor/env/copilot.sh

# Verify the env is active
echo $HTTPS_PROXY        # → http://localhost:8081
echo $NODE_EXTRA_CA_CERTS  # → /Users/<you>/.isartor/ca/isartor-ca.pem
```

### Step 3 — Use Copilot CLI (same shell)

```bash
# Ask Copilot a question — traffic will route through Isartor proxy
gh copilot suggest "list all files in a directory"

# Or explain
gh copilot explain "what does git rebase do"
```

### Step 4 — Verify Traffic Hit Isartor

```bash
# Check proxy recent decisions
./target/release/isartor connect status \
  --gateway-url http://localhost:8080 \
  --gateway-api-key changeme

# Check prompt stats
./target/release/isartor stats \
  --gateway-url http://localhost:8080 \
  --gateway-api-key changeme
```

You should see `proxy_recent_requests > 0` and Copilot entries in **By Client**.

### Step 5 — Ask Repeated Questions (cache test)

```bash
# Ask the same thing twice — second hit should be L1a
gh copilot suggest "list all files in a directory"
gh copilot suggest "list all files in a directory"

# Check stats — deflected count should have increased
./target/release/isartor stats \
  --gateway-url http://localhost:8080 \
  --gateway-api-key changeme
```

### Disconnect

```bash
./target/release/isartor connect copilot --disconnect
# then unset in your shell:
unset HTTPS_PROXY NODE_EXTRA_CA_CERTS ISARTOR_COPILOT_ENABLED
```

---

## Feature Coverage Matrix

| Feature | Test | Section |
|---|---|---|
| Health endpoint | `curl /health` | §2 |
| Liveness probe | `curl /healthz` | §2 |
| OpenAI `/v1/chat/completions` | curl + jq | §3 |
| Anthropic `/v1/messages` | curl + jq | §4 |
| Native `/api/chat` | curl + jq | §5 |
| L1a exact-cache deflection | repeated prompt | §6 |
| L1b semantic-cache deflection | paraphrased prompt | §7 |
| Auth rejection | no X-API-Key | §8 |
| Prompt stats endpoint | `/debug/stats/prompts` | §9 |
| isartor stats CLI | `isartor stats` | §9 |
| Proxy decisions endpoint | `/debug/proxy/recent` | §10 |
| Connect status CLI | `isartor connect status` | §11 |
| Built-in demo | `isartor demo` | §12 |
| Copilot CLI proxy routing | source env + gh copilot | Copilot CLI |
| Cache hit via Copilot | repeated gh copilot | Copilot CLI §5 |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused :8080` | Server not started | Run `./target/release/isartor up` |
| `isartor update` fails after stop | Stale `HTTPS_PROXY` in shell | `unset HTTPS_PROXY HTTP_PROXY` |
| Copilot traffic not showing in stats | Wrong shell / env not sourced | `source ~/.isartor/env/copilot.sh` then restart Copilot CLI |
| L1b miss on paraphrase | Semantic index cold | Send several prompts first to warm the index |
| `l3: no_api_key` in health | No LLM key set | Set `ISARTOR__EXTERNAL_LLM_API_KEY` or use cache/demo mode |

See also: [Troubleshooting](troubleshooting.md) · [Contributing](contributing.md)

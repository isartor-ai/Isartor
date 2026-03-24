# OpenAI Codex CLI

OpenAI Codex CLI integrates via `OPENAI_BASE_URL`, routing requests through
Isartor's OpenAI-compatible `/v1` surface, including `/v1/chat/completions` and
`/v1/models`.

## Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure Codex
isartor connect codex

# 3. Source the env file
source ~/.isartor/env/codex.sh

# 4. Run Codex
codex --model o3-mini
```

## How it works

1. `isartor connect codex` writes `OPENAI_BASE_URL` and `OPENAI_API_KEY` to `~/.isartor/env/codex.sh`
2. Codex can query `/v1/models` to discover the configured model
3. Codex sends chat requests to Isartor's `/v1/chat/completions` endpoint
4. Isartor supports OpenAI streaming SSE and tool-call passthrough for compatible agent workflows
5. Isartor forwards to the configured upstream as Layer 3 when not deflected
6. Use `--model` to select any model name configured in your L3 provider

## Disconnecting

```bash
isartor connect codex --disconnect
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Codex not routing through Isartor | Env vars not loaded | Run `source ~/.isartor/env/codex.sh` in your shell |
| Codex cannot list models | `/v1/models` unreachable or auth mismatch | Test `curl http://localhost:8080/v1/models` with the same auth settings |

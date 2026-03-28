# Quick Start

This guide walks you through starting Isartor, making your first request, observing a cache hit, and checking stats. If you haven't installed Isartor yet, see the [Installation](./installation.md) guide.

## Guided Setup

For the smoothest first-run experience, use the setup wizard:

```bash
isartor setup
```

The wizard can:

- choose your Layer 3 provider
- collect the provider API key and model
- optionally set the Isartor gateway API key
- configure Layer 2 as disabled, embedded, or sidecar
- connect one or more tools
- run a final verification pass

The older explicit commands still work if you prefer scripting or manual control.

## Starting Isartor

```bash
isartor up           # start the API gateway only
isartor up --detach  # start in background and return to the shell
isartor up copilot   # start gateway + CONNECT proxy for Copilot CLI
```

Other useful commands:

```bash
isartor init         # generate a commented config scaffold
isartor setup        # guided setup for provider, L2, connectors, and verification
isartor set-key -p openai  # configure your LLM provider API key
isartor check        # verify provider/model/key masking and live connectivity
isartor demo         # run the post-install showcase
isartor stop         # stop a running Isartor instance (uses PID file)
isartor update       # self-update to the latest version from GitHub releases
```

## Making Your First Request

Isartor exposes an OpenAI-compatible API. Send a request to the `/v1/chat/completions` endpoint:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-2-2b-it",
    "messages": [
      {"role": "user", "content": "Explain the quantum Hall effect in detail, including its significance for condensed matter physics and any applications in modern technology."}
    ]
  }'
```

**Expected JSON Response (snippet):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The quantum Hall effect is a phenomenon..."
      }
    }
  ],
  "usage": { ... }
}
```

**Console Log (snippet):**
```
INFO  [slm_triage] Layer 3 fallback: OpenAI
INFO  [cache] Layer 1a miss: quantum Hall effect prompt
```

The first request is a cache miss — Layer 2 triages it and Layer 3 routes it to your configured cloud provider.

OpenAI-compatible clients can also:

- call `GET /v1/models` to discover the configured model
- send `"stream": true` and receive OpenAI-style SSE responses
- use tool/function calling fields such as `tools`, `tool_choice`, and `functions`

You can also use the native API:

```bash
curl -s http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Calculate 2+2"}'
```

## Seeing a Cache Hit

Repeat the same request:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-2-2b-it",
    "messages": [
      {"role": "user", "content": "Explain the quantum Hall effect in detail, including its significance for condensed matter physics and any applications in modern technology."}
    ]
  }'
```

**Expected JSON Response (snippet):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The quantum Hall effect is a phenomenon..."
      }
    }
  ],
  "usage": { ... }
}
```

**Console Log (snippet):**
```
INFO  [cache] Layer 1a exact match: quantum Hall effect prompt
INFO  [slm_triage] Short-circuit: cache hit
```

This time the response comes from the Layer 1a exact cache — sub-millisecond, zero tokens consumed, no cloud call.

## Checking Stats

View prompt totals, layer hit rates, and recent routing history:

```bash
isartor stats
```

## Connecting an AI Tool

Isartor works as a drop-in replacement for any OpenAI-compatible client. Point your favourite AI tool at `http://localhost:8080/v1` and it will route through the Deflection Stack automatically.

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="your-api-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Summarise this document."}],
)
```

If your client probes models first, this also works:

```bash
curl -sS http://localhost:8080/v1/models
```

For detailed setup guides for GitHub Copilot CLI, Claude Code, Cursor, and other tools, see the [Integrations](../integrations/overview.md) section.

---

For advanced configuration, see the [Configuration Reference](../configuration/reference.md) and [Architecture](../concepts/architecture.md).

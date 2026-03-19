# Isartor Integration Guide

Isartor is an **OpenAI-compatible and Anthropic-compatible gateway** that deflects
repeated or simple prompts at Layer 1 (cache) and Layer 2 (local SLM) before they
reach the cloud. Clients integrate by **overriding their base URL** to point at
Isartor or by using **preToolUse hooks** — no proxy, no MITM, no CA certificates.

## Endpoints

Isartor's server defaults to: `http://localhost:8080`.

Authenticated chat endpoints:

- **Native Isartor** (recommended for direct use)
  - `POST /api/chat`
  - `POST /api/v1/chat` (alias)
- **OpenAI Chat Completions compatible**
  - `POST /v1/chat/completions`
- **Anthropic Messages compatible**
  - `POST /v1/messages`
- **Copilot preToolUse hook**
  - `POST /api/v1/hook/pretooluse`

## Authentication

Isartor can enforce a gateway key on authenticated routes when Layer 0 auth is enabled.

Supported headers:

- `X-API-Key: <gateway_api_key>`
- `Authorization: Bearer <gateway_api_key>` (useful for OpenAI/Anthropic-compatible clients)

By default, `gateway_api_key` is empty and **auth is disabled** (local-first). To enable gateway authentication, set `ISARTOR__GATEWAY_API_KEY` to a secret value. In production, **always** set a strong key.

## Observability headers

All endpoints in the Deflection Stack include:

- `X-Isartor-Layer`: `l1a` | `l1b` | `l2` | `l3` | `l0`
- `X-Isartor-Deflected`: `true` if resolved locally (no cloud call)

## Example: OpenAI-compatible request

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "2 + 2?"}
    ]
  }'
```

If gateway auth is enabled, also add:

```bash
-H 'Authorization: Bearer your-secret-key'
```

## Example: Anthropic-compatible request

```bash
curl -sS http://localhost:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet-4-6",
    "system": "Be concise.",
    "max_tokens": 100,
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "What is the capital of France?"}]
      }
    ]
  }'
```

If gateway auth is enabled, also add:

```bash
-H 'X-API-Key: your-secret-key'
```

## Client integrations: `isartor connect …`

Isartor ships a helper CLI to configure popular clients to route through the
gateway. Each integration uses the lightest-weight mechanism available for that
client — either a **base URL override** or a **preToolUse hook**.

```bash
# Show what's connected and test the gateway
isartor connect status

# GitHub Copilot CLI (preToolUse hook)
isartor connect copilot

# Claude Code (base URL override)
isartor connect claude

# Antigravity (base URL override)
isartor connect antigravity

# OpenClaw (provider base URL)
isartor connect openclaw
```

Add `--gateway-api-key <key>` to these commands only if you have explicitly enabled gateway auth.

### GitHub Copilot CLI (preToolUse hook)

Copilot CLI integrates via a **preToolUse hook** that sends tool-call metadata to
Isartor before execution. Isartor can deflect the call at L1/L2 or let it pass
through to the Copilot upstream as Layer 3.

#### Prerequisites

- Isartor installed (`curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh`)
- GitHub Copilot CLI installed (`gh extension install github/gh-copilot`)

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure Copilot hooks
isartor connect copilot

# 3. Register the hook with Copilot CLI (shown in output)
# Follow instructions printed by the connect command

# 4. Use Copilot normally
gh copilot suggest "explain this function"
```

#### How it works

1. `isartor connect copilot` generates a hook script at `~/.isartor/hooks/copilot_pretooluse.sh`
2. The hook script POSTs tool-call metadata to Isartor's `/api/v1/hook/pretooluse` endpoint
3. Isartor evaluates the request through the Deflection Stack (L1 → L2 → L3)
4. If deflected at L1 or L2, the cached/local response is returned without a cloud call

#### Disconnecting

```bash
isartor connect copilot --disconnect
```

### Claude Code (base URL override)

Claude Code integrates via `ANTHROPIC_BASE_URL`, pointing all API traffic at
Isartor's `/v1/messages` endpoint.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure Claude Code
isartor connect claude

# 3. Claude Code now routes through Isartor automatically
```

#### How it works

1. `isartor connect claude` sets `ANTHROPIC_BASE_URL` in `~/.claude/settings.json`
2. Claude Code sends requests to Isartor's `/v1/messages` endpoint
3. Isartor forwards to the Anthropic API as Layer 3 when the request is not deflected

#### Disconnecting

```bash
isartor connect claude --disconnect
```

### Antigravity (base URL override)

Antigravity integrates via an environment file that sets the OpenAI base URL.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure Antigravity
isartor connect antigravity

# 3. Source the env file
source ~/.isartor/env/antigravity.sh
```

#### How it works

1. `isartor connect antigravity` writes `OPENAI_BASE_URL` and `OPENAI_API_KEY` to `~/.isartor/env/antigravity.sh`
2. Antigravity sends requests to Isartor's `/v1/chat/completions` endpoint
3. Isartor forwards to the OpenAI-compatible upstream as Layer 3 when the request is not deflected

#### Disconnecting

```bash
isartor connect antigravity --disconnect
```

### OpenClaw (provider base URL)

OpenClaw integrates via a JSON patch to its provider configuration.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure OpenClaw
isartor connect openclaw
```

#### How it works

1. `isartor connect openclaw` patches OpenClaw's provider config to set the base URL to Isartor
2. OpenClaw sends requests to Isartor's gateway
3. Isartor forwards to the configured upstream as Layer 3 when the request is not deflected

#### Disconnecting

```bash
isartor connect openclaw --disconnect
```

## Connection status

```bash
# Check all connected clients
isartor connect status
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "connection refused" | Isartor not running | Run `isartor up` first |
| Copilot works but bypasses Isartor | Hook not registered | Run `isartor connect copilot` and follow registration instructions |
| Claude not routing through Isartor | `settings.json` not updated | Run `isartor connect claude` |
| Gateway returns 401 | Auth enabled but key not configured | Add `--gateway-api-key` to connect command |

---

For more details, see [README.md](../README.md) and [docs/2-ARCHITECTURE.md](2-ARCHITECTURE.md).

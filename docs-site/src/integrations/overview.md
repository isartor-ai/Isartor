# AI Tool Integrations

Isartor is an **OpenAI-compatible and Anthropic-compatible gateway** that deflects
repeated or simple prompts at Layer 1 (cache) and Layer 2 (local SLM) before they
reach the cloud. Clients integrate by **overriding their base URL** to point at
Isartor or by registering Isartor as an **MCP server** — no proxy, no MITM, no CA
certificates.

## Endpoints

Isartor's server defaults to: `http://localhost:8080`.

Authenticated chat endpoints:

| Endpoint | Protocol | Path |
|----------|----------|------|
| **Native Isartor** (recommended for direct use) | Native | `POST /api/chat` / `POST /api/v1/chat` |
| **OpenAI Models** | OpenAI | `GET /v1/models` |
| **OpenAI Chat Completions** | OpenAI | `POST /v1/chat/completions` |
| **Anthropic Messages** | Anthropic | `POST /v1/messages` |
| **Cache lookup / store** (used by MCP clients) | Native | `POST /api/v1/cache/lookup` / `POST /api/v1/cache/store` |

## Authentication

Isartor can enforce a gateway key on authenticated routes when Layer 0 auth is
enabled.

Supported headers:

- `X-API-Key: <gateway_api_key>`
- `Authorization: Bearer <gateway_api_key>` (useful for OpenAI/Anthropic-compatible clients)

By default, `gateway_api_key` is empty and **auth is disabled** (local-first). To
enable gateway authentication, set `ISARTOR__GATEWAY_API_KEY` to a secret value. In
production, **always** set a strong key.

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

Many OpenAI-compatible SDKs and coding agents also call:

```bash
curl -sS http://localhost:8080/v1/models
```

OpenAI-compatible agent features supported by Isartor:

- `GET /v1/models` for model discovery
- `stream: true` on `/v1/chat/completions` with OpenAI-style SSE and `data: [DONE]`
- `tools`, `tool_choice`, `functions`, and `function_call` passthrough
- `tool_calls` preserved in provider responses
- tool-aware exact cache keys, with semantic cache skipped for tool-use flows

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

## Supported tools at a glance

| Tool | Command | Mechanism |
|------|---------|-----------|
| [GitHub Copilot CLI](copilot.md) | `isartor connect copilot` | MCP server (cache-only) |
| [GitHub Copilot in VS Code](copilot-vscode.md) | `isartor connect copilot-vscode` | Managed `settings.json` debug overrides |
| [OpenClaw](openclaw.md) | `isartor connect openclaw` | Provider patch for `openclaw.json` |
| [OpenCode](opencode.md) | `isartor connect opencode` | Global provider + auth config |
| [Claude Code + GitHub Copilot](claude-copilot.md) | `isartor connect claude-copilot` | Claude base URL override + Copilot-backed L3 |
| [Claude Code](claude-code.md) | `isartor connect claude` | Base URL override |
| [Cursor IDE](cursor.md) | `isartor connect cursor` | Base URL override + MCP |
| [OpenAI Codex CLI](codex.md) | `isartor connect codex` | Base URL override |
| [Gemini CLI](gemini.md) | `isartor connect gemini` | Base URL override |
| [Antigravity](antigravity.md) | `isartor connect antigravity` | Base URL override |
| [Generic / other tools](generic.md) | `isartor connect generic` | Base URL override |

Add `--gateway-api-key <key>` to any connect command only if you have explicitly
enabled gateway auth.

## Connection status

```bash
# Check all connected clients
isartor connect status
```

## Global troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "connection refused" | Isartor not running | Run `isartor up` first |
| Gateway returns 401 | Auth enabled but key not configured | Add `--gateway-api-key` to connect command |

For tool-specific troubleshooting, see each integration page above.

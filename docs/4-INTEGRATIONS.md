# Isartor Integration Guide

Isartor is an **OpenAI-compatible and Anthropic-compatible gateway** that deflects
repeated or simple prompts at Layer 1 (cache) and Layer 2 (local SLM) before they
reach the cloud. Clients integrate by **overriding their base URL** to point at
Isartor or by registering Isartor as an **MCP server** — no proxy, no MITM, no CA certificates.

## Endpoints

Isartor's server defaults to: `http://localhost:8080`.

Authenticated chat endpoints:

- **Native Isartor** (recommended for direct use)
  - `POST /api/chat`
  - `POST /api/v1/chat` (alias)
- **OpenAI Chat Completions compatible**
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- **Anthropic Messages compatible**
  - `POST /v1/messages`
- **Cache lookup / store** (used by MCP clients)
  - `POST /api/v1/cache/lookup`
  - `POST /api/v1/cache/store`
- **Copilot preToolUse hook** (legacy)
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

Many OpenAI SDKs also call the model-discovery endpoint automatically:

```bash
curl -sS http://localhost:8080/v1/models
```

OpenAI-compatible agent features supported by Isartor:

- `GET /v1/models` for model discovery
- `stream: true` on `/v1/chat/completions` with OpenAI-style SSE chunks and `data: [DONE]`
- `tools`, `tool_choice`, `functions`, and `function_call` passthrough on tool-enabled requests
- `tool_calls` preserved in provider responses, with tool-aware cache keys and semantic cache disabled for tool-use flows

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
client — an **MCP server** (for Copilot CLI) or a **base URL override**.

```bash
# Show what's connected and test the gateway
isartor connect status

# GitHub Copilot CLI (MCP server — isartor_chat tool)
isartor connect copilot

# GitHub Copilot in VS Code (settings.json debug overrides)
isartor connect copilot-vscode

# Claude Code (base URL override)
isartor connect claude

# Cursor IDE (base URL override + MCP)
isartor connect cursor

# OpenAI Codex CLI (base URL override)
isartor connect codex

# Gemini CLI (base URL override)
isartor connect gemini

# OpenCode (global provider config)
isartor connect opencode

# Antigravity (base URL override)
isartor connect antigravity

# OpenClaw (provider base URL)
isartor connect openclaw

# Any OpenAI-compatible tool (generic connector)
isartor connect generic --tool-name Windsurf --base-url-var OPENAI_BASE_URL --api-key-var OPENAI_API_KEY
```

Add `--gateway-api-key <key>` to these commands only if you have explicitly enabled gateway auth.

### GitHub Copilot CLI (MCP server — cache-only)

Copilot CLI integrates via an **MCP (Model Context Protocol) server** that
Isartor registers as a stdio subprocess. Isartor also exposes the same MCP
tools over **Streamable HTTP** at `http://localhost:8080/mcp/` for editors and
web agents that prefer HTTP/SSE transport. Both transports expose two tools:

- **`isartor_chat`** — cache lookup only. Returns the cached answer on hit
  (L1a exact or L1b semantic), or an empty string on miss. On a miss, Copilot
  uses its own LLM to answer — Isartor never routes through its configured L3
  provider for Copilot traffic.
- **`isartor_cache_store`** — stores a prompt/response pair in Isartor's cache
  so future identical or similar prompts are deflected locally.

This design means Copilot still owns the conversation loop, while Isartor acts
as a transparent cache layer that reduces redundant cloud calls. On a cache hit,
Isartor returns the cached text and **does not call its own Layer 3 provider**.
Copilot CLI may still emit its normal final-answer event after the tool result,
but that is a Copilot-side render step rather than an Isartor L3 forward.

#### Prerequisites

- Isartor installed (`curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh`)
- GitHub Copilot CLI installed

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up --detach

# 2. Register the MCP server with Copilot CLI
isartor connect copilot

# 3. Start Copilot normally — plain chat prompts will use Isartor cache first
copilot
```

#### How it works

1. `isartor connect copilot` adds an `isartor` entry to `~/.copilot/mcp-config.json`
2. `isartor connect copilot` also installs a managed instruction block in `~/.copilot/copilot-instructions.md`
3. When Copilot CLI starts, it launches `isartor mcp` as a stdio subprocess and loads the Isartor instruction block
4. The MCP server exposes `isartor_chat` (cache lookup) and `isartor_cache_store` (cache write)
5. For plain conversational prompts, Copilot now prefers this flow:
   - Call `isartor_chat` with the user's prompt
   - **Cache hit**: return the cached answer immediately, verbatim
   - **Cache miss**: answer with Copilot's own model, then call `isartor_cache_store`
6. When Copilot calls `isartor_chat`:
   - **Cache hit** (L1a exact or L1b semantic): returns the cached answer instantly
   - **Cache miss**: returns empty → Copilot uses its own LLM
7. After Copilot gets an answer from its LLM, it can call `isartor_cache_store` to
   populate the cache for future requests

#### HTTP/SSE MCP endpoint

Isartor now exposes the same MCP tool surface at `/mcp/` using Streamable HTTP:

- `POST /mcp/` — client → server JSON-RPC
- `GET /mcp/` — server → client SSE stream
- `DELETE /mcp/` — explicit session teardown

The HTTP transport uses the MCP `Mcp-Session-Id` header after `initialize`, and
supports both JSON responses and SSE responses for POST requests. A minimal
editor config looks like:

```json
{"servers":{"isartor":{"type":"http","url":"http://localhost:8080/mcp/"}}}
```

#### Important note about "still going to L3"

If you inspect Copilot CLI JSON traces, you may still see a normal
`final_answer` event after `isartor_chat` returns a cache hit. That does **not**
mean Isartor forwarded the prompt to its own Layer 3 provider. The important
signal is Isartor's own log and headers:

- `Cache lookup: L1a exact hit` or `Cache lookup: L1b semantic hit`
- no new `Layer 3: Forwarding to LLM via Rig` entry for that prompt

In other words:

- **Isartor L3 call** = bad for a cache hit
- **Copilot final-answer render after a tool hit** = expected CLI behavior

Isartor now installs stricter Copilot instructions that tell Copilot to emit the
cached tool result verbatim on cache hits, without paraphrasing or extra tool calls.

#### Cache endpoints (used by MCP internally)

The MCP server calls these HTTP endpoints on the Isartor gateway:

```bash
# Cache lookup — returns cached response or 204 No Content
curl -X POST http://localhost:8080/api/v1/cache/lookup \
  -H "Content-Type: application/json" \
  -d '{"prompt": "capital of France"}'

# Cache store — saves a prompt/response pair
curl -X POST http://localhost:8080/api/v1/cache/store \
  -H "Content-Type: application/json" \
  -d '{"prompt": "capital of France", "response": "The capital of France is Paris."}'
```

#### Custom gateway URL

```bash
# If Isartor runs on a non-default port
isartor connect copilot --gateway-url http://localhost:18080
```

#### Disconnecting

```bash
isartor connect copilot --disconnect
```

This removes the `isartor` entry from `~/.copilot/mcp-config.json`.
It also removes the managed Isartor block from `~/.copilot/copilot-instructions.md`.

### Claude Code + GitHub Copilot (experimental)

Use Claude Code with GitHub Copilot as the Layer 3 backend, while Isartor keeps
L1a/L1b cache deflection in front of it:

```bash
isartor connect claude-copilot
isartor stop
isartor up --detach
claude
```

→ Full guide: [CLAUDE-COPILOT.md](CLAUDE-COPILOT.md)

> **Note:** this integration is currently best for plain prompting. Isartor's
> Anthropic compatibility path is still text-oriented, so advanced Anthropic
> tool-use flows may still require follow-up work.

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

### GitHub Copilot in VS Code (settings.json debug overrides)

GitHub Copilot in VS Code can be pointed at Isartor by writing the
`github.copilot.advanced.debug.*` override settings into VS Code's
`settings.json`.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up --detach

# 2. Configure VS Code automatically
isartor connect copilot-vscode

# 3. Reload VS Code
# Developer: Reload Window
```

#### How it works

1. `isartor connect copilot-vscode` auto-detects your VS Code user `settings.json`
2. It backs up the existing file to `settings.json.isartor-backup`
3. It writes:
   - `github.copilot.advanced.debug.overrideProxyUrl`
   - `github.copilot.advanced.debug.overrideCAPIUrl`
   - `github.copilot.advanced.debug.chatOverrideProxyUrl`
4. It validates that Isartor is reachable before writing

> **Caveat:** these `debug.*` settings are unsupported internal Copilot settings
> from GitHub and may break on extension updates.

#### Disconnecting

```bash
isartor connect copilot-vscode --disconnect
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

### Cursor IDE (base URL override + MCP)

Cursor IDE integrates via the **OpenAI Base URL override** in Cursor's model
settings, and optionally via **MCP server registration** for tool-based
integration.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure Cursor
isartor connect cursor

# 3. Open Cursor → Settings → Cursor Settings → Models
# 4. Enable "Override OpenAI Base URL" and enter: http://localhost:8080/v1
# 5. Paste the API key shown in the connect output
# 6. Add a custom model name (e.g. gpt-4o) and enable it
# 7. Use Ask or Plan mode (Agent mode doesn't support custom keys yet)
```

#### How it works

1. `isartor connect cursor` writes a reference env file to `~/.isartor/env/cursor.sh`
2. It also registers Isartor as an MCP server in `~/.cursor/mcp.json`
3. In Cursor, override the OpenAI Base URL to point at Isartor's `/v1` endpoint
4. The generated MCP entry points at Isartor's HTTP/SSE MCP endpoint: `http://localhost:8080/mcp/`
5. All chat completions requests route through Isartor's L1/L2/L3 deflection stack
6. Cursor's Ask and Plan modes are supported; Agent mode requires native keys

#### Disconnecting

```bash
isartor connect cursor --disconnect
```

### OpenAI Codex CLI (base URL override)

OpenAI Codex CLI integrates via `OPENAI_BASE_URL`, routing all requests through
Isartor's OpenAI-compatible `/v1/chat/completions` endpoint.

#### Step-by-step setup

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

#### How it works

1. `isartor connect codex` writes `OPENAI_BASE_URL` and `OPENAI_API_KEY` to `~/.isartor/env/codex.sh`
2. Codex sends requests to Isartor's `/v1/chat/completions` endpoint
3. Isartor forwards to the configured upstream as Layer 3 when not deflected
4. Use `--model` to select any model name configured in your L3 provider

#### Disconnecting

```bash
isartor connect codex --disconnect
```

### Gemini CLI (base URL override)

Gemini CLI integrates via `GEMINI_API_BASE_URL`, routing requests through
Isartor's gateway.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure Gemini CLI
isartor connect gemini

# 3. Source the env file
source ~/.isartor/env/gemini.sh

# 4. Run Gemini CLI
gemini
```

#### How it works

1. `isartor connect gemini` writes `GEMINI_API_BASE_URL` and `GEMINI_API_KEY` to `~/.isartor/env/gemini.sh`
2. Gemini CLI sends requests to Isartor's gateway
3. Isartor forwards to the configured upstream as Layer 3 when not deflected

#### Disconnecting

```bash
isartor connect gemini --disconnect
```

### OpenCode (global provider config)

OpenCode reads a global JSON config and auth store. Isartor installs an
`isartor` provider backed by `@ai-sdk/openai-compatible` and points it at the
gateway's `/v1` endpoint.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure OpenCode
isartor connect opencode

# 3. Start OpenCode
opencode
```

#### How it works

1. `isartor connect opencode` backs up `~/.config/opencode/opencode.json` to `~/.config/opencode/opencode.json.isartor-backup`
2. It writes an `isartor` provider to `~/.config/opencode/opencode.json`
3. It writes a matching auth entry to `~/.local/share/opencode/auth.json`
4. The provider uses `@ai-sdk/openai-compatible` with `baseURL` set to `http://localhost:8080/v1` (or your configured `--gateway-url`)
5. The auth entry uses your gateway API key when gateway auth is enabled, otherwise a dummy `isartor-local` key

#### Disconnecting

```bash
isartor connect opencode --disconnect
```

Disconnect restores the pre-Isartor config/auth backups when available, and
otherwise removes only the managed `isartor` entries.

### Generic (any OpenAI-compatible tool)

For tools not explicitly supported (Windsurf, Zed, Cline, Roo Code, etc.),
use the generic connector to generate an env script that sets the tool's
base URL environment variable to point at Isartor.

#### Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure the tool (example: Windsurf)
isartor connect generic \
  --tool-name Windsurf \
  --base-url-var OPENAI_BASE_URL \
  --api-key-var OPENAI_API_KEY

# 3. Source the env file
source ~/.isartor/env/windsurf.sh

# 4. Start the tool
```

#### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--tool-name` | yes | Display name (also used for env script filename) |
| `--base-url-var` | yes | Env var the tool reads for its API base URL |
| `--api-key-var` | no | Env var the tool reads for its API key |
| `--no-append-v1` | no | Don't append `/v1` to the gateway URL |

#### Disconnecting

```bash
isartor connect generic \
  --tool-name Windsurf \
  --base-url-var OPENAI_BASE_URL \
  --disconnect
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
| Copilot has no `isartor_chat` tool | MCP server not registered | Run `isartor connect copilot` |
| Copilot works but bypasses cache | Isartor instructions not installed or custom instructions disabled | Run `isartor connect copilot` again and do not launch Copilot with `--no-custom-instructions` |
| VS Code Copilot not routing through Isartor | `github.copilot.advanced` overrides missing or stale | Run `isartor connect copilot-vscode`, then reload VS Code |
| Cache never hits for Copilot | Responses not stored after LLM answers | Ask Copilot to call `isartor_cache_store` after answering |
| Claude not routing through Isartor | `settings.json` not updated | Run `isartor connect claude` |
| Cursor not routing through Isartor | Base URL override not set | Open Cursor Settings → Models → enable Override OpenAI Base URL |
| Codex/Gemini not routing | Env vars not loaded | Run `source ~/.isartor/env/<tool>.sh` in your shell |
| Gateway returns 401 | Auth enabled but key not configured | Add `--gateway-api-key` to connect command |

---

For more details, see [README.md](../README.md) and [docs/2-ARCHITECTURE.md](2-ARCHITECTURE.md).

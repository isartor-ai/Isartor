# GitHub Copilot CLI

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

## Prerequisites

- Isartor installed (`curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh`)
- GitHub Copilot CLI installed

## Step-by-step setup

```bash
# 1. Start Isartor
isartor up --detach

# 2. Register the MCP server with Copilot CLI
isartor connect copilot

# 3. Start Copilot normally — plain chat prompts will use Isartor cache first
copilot
```

## How it works

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

## HTTP/SSE MCP endpoint

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

## Important note about "still going to L3"

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

## Cache endpoints (used by MCP internally)

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

## Custom gateway URL

```bash
# If Isartor runs on a non-default port
isartor connect copilot --gateway-url http://localhost:18080
```

## Disconnecting

```bash
isartor connect copilot --disconnect
```

This removes the `isartor` entry from `~/.copilot/mcp-config.json`.
It also removes the managed Isartor block from `~/.copilot/copilot-instructions.md`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Copilot has no `isartor_chat` tool | MCP server not registered | Run `isartor connect copilot` |
| Copilot works but bypasses cache | Isartor instructions not installed or custom instructions disabled | Run `isartor connect copilot` again and do not launch Copilot with `--no-custom-instructions` |
| Cache never hits for Copilot | Responses not stored after LLM answers | Ask Copilot to call `isartor_cache_store` after answering |

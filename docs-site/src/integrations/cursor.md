# Cursor IDE

Cursor IDE integrates via the **OpenAI Base URL override** in Cursor's model
settings, and optionally via **MCP server registration** for tool-based
integration.

## Step-by-step setup

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

## How it works

1. `isartor connect cursor` writes a reference env file to `~/.isartor/env/cursor.sh`
2. It also registers Isartor as an MCP server in `~/.cursor/mcp.json`
3. In Cursor, override the OpenAI Base URL to point at Isartor's `/v1` endpoint
4. Cursor can use Isartor's `GET /v1/models` endpoint to discover the configured model
5. All chat completions requests route through Isartor's L1/L2/L3 deflection stack
6. Isartor supports OpenAI streaming SSE, tool-call passthrough, and HTTP/SSE MCP at `http://localhost:8080/mcp/` for compatible Cursor workflows
7. Cursor's Ask and Plan modes are supported; Agent mode requires native keys

Cursor's generated MCP config points at:

```json
{"mcpServers":{"isartor":{"type":"http","url":"http://localhost:8080/mcp/"}}}
```

## Disconnecting

```bash
isartor connect cursor --disconnect
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cursor not routing through Isartor | Base URL override not set | Open Cursor Settings → Models → enable Override OpenAI Base URL |
| Cursor model picker is empty | Cursor cannot reach model discovery | Verify `http://localhost:8080/v1/models` is reachable from Cursor |

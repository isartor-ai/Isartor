# Generic Connector

For tools not explicitly supported, use the generic connector to generate an env
script that sets the tool's base URL environment variable to point at Isartor.

## Compatible tools

The generic connector works with any OpenAI-compatible tool, including:

- **Windsurf**
- **Zed**
- **Cline**
- **Roo Code**
- **Aider**
- **Continue**
- **Antigravity** (also available via `isartor connect antigravity`)
- **OpenClaw** (also available via `isartor connect openclaw`)
- Any other tool that reads an `OPENAI_BASE_URL` or similar environment variable

OpenAI-compatible features exposed by Isartor include:

- `GET /v1/models` for model discovery
- `POST /v1/chat/completions`
- `stream: true` SSE responses
- tool/function calling passthrough (`tools`, `tool_choice`, `functions`, `tool_calls`)

## Step-by-step setup

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

## Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--tool-name` | yes | Display name (also used for env script filename) |
| `--base-url-var` | yes | Env var the tool reads for its API base URL |
| `--api-key-var` | no | Env var the tool reads for its API key |
| `--no-append-v1` | no | Don't append `/v1` to the gateway URL |

## Disconnecting

```bash
isartor connect generic \
  --tool-name Windsurf \
  --base-url-var OPENAI_BASE_URL \
  --disconnect
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Tool not routing through Isartor | Env vars not loaded | Run `source ~/.isartor/env/<tool>.sh` in your shell |
| Tool says no models are available | It expects OpenAI model discovery | Verify it can reach `http://localhost:8080/v1/models` |

# GitHub Copilot in VS Code

Route GitHub Copilot's code completions and chat requests in VS Code through
Isartor, so repetitive prompts are deflected locally via the L1a/L1b cache
layers. This reduces cloud API calls, lowers latency for repeated patterns,
and gives you per-tool visibility in `isartor stats`.

> **How is this different from Copilot CLI?**
> The [Copilot CLI integration](copilot.md) uses an MCP server for the
> terminal-based `copilot` command. This page covers **VS Code** — the editor
> extension that provides inline completions and Copilot Chat.

---

## Prerequisites

- **Isartor** installed and running (`isartor up --detach`)
- **GitHub Copilot** VS Code extension installed (requires a Copilot subscription)
- An **LLM provider API key** configured in Isartor for Layer 3 fallback
  (`isartor set-key -p openai` or similar)

## Step 1 — Start Isartor

```bash
# Install (if not already)
curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh

# Configure your LLM provider key (OpenAI, Anthropic, Azure, etc.)
isartor set-key -p openai

# Start the gateway in the background
isartor up --detach
```

Verify it's running:

```bash
curl http://localhost:8080/health
# {"status":"ok", ...}
```

## Step 2 — Configure VS Code

Recommended:

```bash
isartor connect copilot-vscode
```

This command:

- auto-detects the VS Code `settings.json` path on macOS, Linux, and Windows
- backs up the original file to `settings.json.isartor-backup`
- writes the three `github.copilot.advanced.debug.*` overrides
- refuses to write if Isartor is not reachable

Manual alternative: open your VS Code **User Settings (JSON)** and add:

```json
{
  "github.copilot.advanced": {
    "debug.overrideProxyUrl": "http://localhost:8080",
    "debug.overrideCAPIUrl": "http://localhost:8080/v1",
    "debug.chatOverrideProxyUrl": "http://localhost:8080/v1/chat/completions"
  }
}
```

| Setting | What It Does |
|---------|-------------|
| `debug.overrideProxyUrl` | Routes Copilot's main API traffic through Isartor |
| `debug.overrideCAPIUrl` | Overrides the completions API endpoint (inline suggestions) |
| `debug.chatOverrideProxyUrl` | Overrides the Copilot Chat endpoint |

> **Custom port?** If Isartor runs on a different port, replace `8080`
> with your port everywhere above.

## Step 3 — Restart VS Code

Close and reopen VS Code (or run **"Developer: Reload Window"** from the
command palette). Copilot will now route requests through Isartor.

## Step 4 — Verify

Open any code file and trigger a Copilot suggestion (start typing a comment
or function). Then check Isartor's stats:

```bash
isartor stats
```

You should see requests flowing through Isartor's layers. Repeat the same
prompt and you'll see **L1a cache hits** — Isartor deflected the duplicate
without a cloud call.

For per-tool breakdown:

```bash
isartor stats --by-tool
```

Copilot VS Code traffic appears as `copilot` in the tool column (identified
from the `User-Agent` header). The table now includes requests, cache
hits/misses, average latency, retries, errors, and L1a/L1b safety.

---

## How It Works

```text
VS Code Copilot Extension
        │
        ▼ (HTTP request to overrideProxyUrl)
   ┌─────────────┐
   │   Isartor    │
   │  Gateway     │
   │              │
   │  L1a ──► L1b ──► L3 (Cloud)
   │  hit?    hit?    forward
   └─────────────┘
        │
        ▼
   Response back to VS Code
```

1. Copilot sends completion/chat requests to Isartor instead of GitHub's servers
2. **L1a Exact Cache** — sub-millisecond hit for identical prompts (< 1 ms)
3. **L1b Semantic Cache** — catches variations of the same prompt (1–5 ms)
4. **L3 Cloud** — only genuinely new prompts reach your configured LLM provider
5. Response flows back to Copilot transparently — no change to the editor UX

## Disconnecting

```bash
isartor connect copilot-vscode --disconnect
```

If a backup exists, Isartor restores it. Otherwise it removes only the three
managed `github.copilot.advanced.debug.*` keys.

## Benefits

| Benefit | How |
|---------|-----|
| **Reduced API costs** | Repetitive completions are served from cache |
| **Lower latency** | Cache hits return in < 5 ms vs hundreds of ms for cloud |
| **Visibility** | `isartor stats --by-tool` shows Copilot request counts, cache hit/miss safety, latency, retries, and errors |
| **Privacy** | Cached prompts never leave your machine on repeat requests |
| **Model flexibility** | Route L3 to any provider (OpenAI, Anthropic, Azure, local Ollama) |

---

## Advanced Configuration

### Use a specific LLM provider for Layer 3

Isartor routes surviving (non-cached) prompts to your configured L3 provider.
You can use any supported provider:

```bash
# OpenAI (default)
isartor set-key -p openai

# Anthropic
isartor set-key -p anthropic

# Azure OpenAI
export ISARTOR__LLM_PROVIDER=azure
export ISARTOR__EXTERNAL_LLM_URL=https://<resource>.openai.azure.com
export ISARTOR__AZURE_DEPLOYMENT_ID=<deployment>
isartor set-key -p azure
```

### Adjust cache sensitivity

Tune the semantic cache threshold to control how similar a prompt must be to
trigger an L1b hit:

```bash
# Default: 0.92 (higher = stricter matching)
export ISARTOR__SIMILARITY_THRESHOLD=0.90
```

See the [Configuration Reference](../configuration/reference.md) for all
available options.

### Enable monitoring

```bash
export ISARTOR__ENABLE_MONITORING=true
export ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317
```

See [Metrics & Tracing](../observability/metrics-tracing.md) for Grafana
dashboards and OTel setup.

---

## Known Limitations

1. **Copilot Chat override** — The `debug.chatOverrideProxyUrl` setting may
   not be fully respected by all versions of the Copilot Chat extension
   ([tracking issue](https://github.com/microsoft/vscode-copilot-release/issues/7802)).
   Inline code completions (`debug.overrideCAPIUrl`) work reliably. If chat
   requests bypass Isartor, try using the global VS Code proxy setting as a
   workaround:

   ```json
   {
     "http.proxy": "http://localhost:8080"
   }
   ```

   > **Note:** This routes _all_ VS Code HTTP traffic through Isartor, not
   > just Copilot. Use a PAC script if you need finer control.

2. **Authentication** — These `debug.*` settings bypass Copilot's normal
   GitHub authentication. Isartor handles the LLM provider auth via its own
   API key configuration. Your Copilot subscription is still required for the
   extension to activate.

3. **Extension updates** — VS Code may update the Copilot extension
   automatically. If the proxy stops working after an update, verify the
   settings are still present in `settings.json` and restart VS Code.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Copilot suggestions stop working | Isartor not running | Run `isartor up --detach` and verify with `curl http://localhost:8080/health` |
| `isartor connect copilot-vscode` cannot find VS Code settings | Non-standard editor config path | Pass through manual JSON editing as a fallback |
| No requests in `isartor stats` | Settings not applied | Verify `settings.json` has the override block, then reload VS Code |
| Chat works but completions don't | Wrong endpoint URL | Ensure `debug.overrideCAPIUrl` ends with `/v1` |
| Completions work but chat doesn't | Known chat override limitation | Add `debug.chatOverrideProxyUrl` or use `http.proxy` as workaround |
| Auth errors from Copilot | Missing L3 provider key | Run `isartor set-key -p openai` (or your provider) |
| High latency on first request | Model loading | First request downloads the embedding model (~25 MB); subsequent requests are fast |

## Reverting

To stop routing Copilot through Isartor, remove the `github.copilot.advanced`
block from your `settings.json` and reload VS Code:

```json
// Remove this entire block:
"github.copilot.advanced": {
    "debug.overrideProxyUrl": "http://localhost:8080",
    "debug.overrideCAPIUrl": "http://localhost:8080/v1",
    "debug.chatOverrideProxyUrl": "http://localhost:8080/v1/chat/completions"
}
```

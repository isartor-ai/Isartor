# OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is a personal AI assistant that
runs on your own devices and connects to channels like WhatsApp, Telegram, Slack,
Discord, and more. Isartor integrates as an OpenAI-compatible provider so all of
OpenClaw's LLM calls pass through the Deflection Stack — cache hits are resolved
locally and context is compressed before reaching the cloud.

## Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Generate the OpenClaw provider patch
isartor connect openclaw

# 3. Apply the patch to your openclaw.json (see below)

# 4. Restart OpenClaw
openclaw gateway --port 18789
```

## How it works

1. `isartor connect openclaw` auto-detects your `openclaw.json` at:
   - `~/.openclaw/openclaw.json` (default)
   - `./openclaw.json` (current directory)
   - or a custom path via `--config-path`
2. It generates a JSON5 provider block and writes it to
   `~/.isartor/patches/openclaw.patch.json5`
3. You paste the block into the `models.providers` section of your `openclaw.json`
4. OpenClaw then routes LLM calls through Isartor as an OpenAI-compatible provider

## Applying the patch

After running `isartor connect openclaw`, add the following to the
`models.providers` block in your `openclaw.json`:

```json5
"isartor": {
  baseUrl: "http://localhost:8080/v1",
  apiKey: "isartor-local",
  api: "openai-chat",
},
```

Then set your agent model to use the `isartor/` prefix:

```json5
agent: {
  model: {
    primary: "isartor/gpt-4o",
    fallbacks: ["isartor/claude-sonnet-4-6", "isartor/groq/llama-3.1-8b-instant"]
  }
}
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o` | Primary model routed through Isartor |
| `--fallbacks` | `claude-sonnet-4-6,groq/llama-3.1-8b-instant` | Comma-separated fallback models |
| `--config-path` | auto-detected | Path to `openclaw.json` |
| `--gateway-api-key` | (none) | Gateway key if auth is enabled |

## Files written

- `~/.isartor/patches/openclaw.patch.json5` — provider block to paste into `openclaw.json`

Backups (if `openclaw.json` is found):

- `openclaw.json.isartor-backup`

## Disconnecting

```bash
isartor connect openclaw --disconnect
```

This removes the patch file. Restore your original `openclaw.json` from the backup
if needed.

## What Isartor does for OpenClaw

| Benefit | How |
|---------|-----|
| **Cache agent loops** | OpenClaw's agent sends the same system instructions + context every turn. L1a exact cache traps repeated prompts instantly. |
| **Semantic dedup** | L1b catches paraphrased follow-ups ("summarize this" ≈ "give me a summary") without an extra cloud call. |
| **Context compression** | L2.5 deduplicates and minifies repeated instruction payloads per session, reducing input tokens sent to the cloud. |
| **Observability** | `isartor stats --by-tool` shows OpenClaw-specific cache hits, latency, and token savings. |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| OpenClaw cannot reach the provider | Isartor not running | Run `isartor up` first |
| Model not found | Missing `isartor/` prefix | Use `isartor/gpt-4o`, not `gpt-4o` |
| Auth errors (401) | Gateway auth enabled | Re-run with `--gateway-api-key` or set `ISARTOR__GATEWAY_API_KEY` |
| Patch file not found | First run needed | Run `isartor connect openclaw` |

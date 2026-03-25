# Antigravity

Antigravity integrates via an OpenAI-compatible base URL override. Isartor
generates a shell env file that sets `OPENAI_BASE_URL` and `OPENAI_API_KEY`
to route all LLM calls through the Deflection Stack.

## Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Generate the env file
isartor connect antigravity

# 3. Activate the environment
source ~/.isartor/env/antigravity.sh

# 4. Start Antigravity
# (it will now use Isartor as its OpenAI endpoint)
```

## How it works

1. `isartor connect antigravity` creates `~/.isartor/env/antigravity.sh`
2. The file exports `OPENAI_BASE_URL` pointing at `http://localhost:8080/v1`
3. It exports `OPENAI_API_KEY` with your gateway key (or a local placeholder)
4. When sourced, Antigravity sends all OpenAI-compatible calls through Isartor

## Files written

- `~/.isartor/env/antigravity.sh`

## Disconnecting

```bash
isartor connect antigravity --disconnect
```

Then restart your shell to clear the exported variables.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused | Isartor not running | Run `isartor up` first |
| Auth errors (401) | Gateway auth enabled | Re-run with `--gateway-api-key` |
| Env not applied | Shell not sourced | Run `source ~/.isartor/env/antigravity.sh` |

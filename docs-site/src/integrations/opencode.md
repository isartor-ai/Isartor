# OpenCode

OpenCode integrates via a global provider config and auth store. Isartor
registers an `isartor` provider backed by `@ai-sdk/openai-compatible` and points
it at the gateway's `/v1` endpoint.

## Step-by-step setup

```bash
# 1. Start Isartor
isartor up

# 2. Configure OpenCode
isartor connect opencode

# 3. Start OpenCode
opencode
```

## How it works

1. `isartor connect opencode` backs up `~/.config/opencode/opencode.json`
2. It writes an `isartor` provider definition to that config file
3. It writes a matching auth entry to `~/.local/share/opencode/auth.json`
4. The provider uses `@ai-sdk/openai-compatible` with `baseURL` set to `http://localhost:8080/v1`
5. If gateway auth is disabled, Isartor writes a dummy local auth key so OpenCode still has a credential to send

## Files written

- `~/.config/opencode/opencode.json`
- `~/.local/share/opencode/auth.json`

Backups:

- `~/.config/opencode/opencode.json.isartor-backup`
- `~/.local/share/opencode/auth.json.isartor-backup`

## Disconnecting

```bash
isartor connect opencode --disconnect
```

Disconnect restores the original files from backup when available. If no backup
exists, it removes only the managed `isartor` entries.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| OpenCode cannot see the Isartor provider | Config file not written | Run `isartor connect opencode` again |
| OpenCode shows auth errors | Gateway auth mismatch | Re-run with `--gateway-api-key` or update `ISARTOR__GATEWAY_API_KEY` |
| OpenCode cannot list models | `/v1/models` unreachable | Verify `curl http://localhost:8080/v1/models` |

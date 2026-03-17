# Isartor Integration Guide

Isartor is a prompt firewall that can act as a **drop-in OpenAI-compatible gateway** (and a minimal Anthropic-compatible gateway) for common SDKs and agent frameworks.

## Endpoints

Isartor’s server defaults to: `http://localhost:8080`.

Authenticated chat endpoints:

- **Native Isartor** (recommended for direct use)
  - `POST /api/chat`
  - `POST /api/v1/chat` (alias)
- **OpenAI Chat Completions compatible**
  - `POST /v1/chat/completions`
- **Anthropic Messages compatible**
  - `POST /v1/messages`

## Authentication

Isartor enforces a gateway key on all authenticated routes.

Supported headers:

- `X-API-Key: <gateway_api_key>`
- `Authorization: Bearer <gateway_api_key>` (useful for OpenAI/Anthropic-compatible clients)

The default key is `changeme`. You should override it via config/env in production.

## Observability headers

All endpoints in the Deflection Stack include:

- `X-Isartor-Layer`: `l1a` | `l1b` | `l2` | `l3` | `l0`
- `X-Isartor-Deflected`: `true` if resolved locally (no cloud call)

## Example: OpenAI-compatible request

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer changeme' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "2 + 2?"}
    ]
  }'
```

## Example: Anthropic-compatible request

```bash
curl -sS http://localhost:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: changeme' \
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

## Client integrations: `isartor connect …`

Isartor ships a helper CLI to configure popular clients to route through the gateway.

```bash
# Show what’s connected and test the gateway
isartor connect status --gateway-url http://localhost:8080 --gateway-api-key changeme

# Claude Code (writes ~/.claude/settings.json)
isartor connect claude --gateway-url http://localhost:8080 --gateway-api-key changeme

# GitHub Copilot CLI (CONNECT proxy + TLS MITM — see below)
isartor connect copilot --gateway-url http://localhost:8080 --gateway-api-key changeme
```

### GitHub Copilot CLI (CONNECT Proxy)

Copilot CLI hard-codes its endpoints and only respects `HTTPS_PROXY`. Isartor intercepts
Copilot traffic using an HTTP CONNECT proxy with TLS MITM:

1. A local CA certificate is generated at `~/.isartor/ca/isartor-ca.pem`
2. The CONNECT proxy runs on `:8081` (configurable via `ISARTOR__PROXY_PORT`)
3. `NODE_EXTRA_CA_CERTS` tells Copilot’s Node.js runtime to trust the local CA

```bash
# Step 1: Start Isartor (runs both API gateway :8080 and CONNECT proxy :8081)
isartor

# Step 2: Configure Copilot CLI
isartor connect copilot

# Step 3: Activate in your shell
source ~/.isartor/env/copilot.sh

# Step 4: Use Copilot CLI normally — traffic routes through Isartor’s Deflection Stack
```

**How it works:**
- The proxy intercepts CONNECT requests to known Copilot domains
- TLS is terminated with a leaf certificate signed by the local CA
- POST requests to `/v1/chat/completions` are routed through the Deflection Stack (L1a → L1b → L3)
- All other traffic is tunnelled transparently

**Intercepted domains:**
- `copilot-proxy.githubusercontent.com`
- `api.github.com`
- `api.individual.githubcopilot.com`
- `api.business.githubcopilot.com`
- `api.enterprise.githubcopilot.com`

Notes:

- The CA is only trusted by Node.js (via `NODE_EXTRA_CA_CERTS`). No system-level trust changes are made.
- Some tools support overriding the OpenAI base URL directly (preferred). Point them at `http://localhost:8080/v1`.

---

For more details, see [README.md](../README.md) and [docs/2-ARCHITECTURE.md](2-ARCHITECTURE.md).

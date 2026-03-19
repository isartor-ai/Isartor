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

By default, `gateway_api_key` is empty and **auth is disabled** (local-first). To enable gateway authentication, set `ISARTOR__GATEWAY_API_KEY` to a secret value. In production, **always** set a strong key.

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

# Claude Code (CONNECT proxy + TLS MITM)
isartor connect claude --gateway-url http://localhost:8080 --gateway-api-key changeme

# GitHub Copilot CLI (CONNECT proxy + TLS MITM — see below)
isartor connect copilot --gateway-url http://localhost:8080 --gateway-api-key changeme

# Antigravity (CONNECT proxy + TLS MITM)
isartor connect antigravity --gateway-url http://localhost:8080 --gateway-api-key changeme
```

### GitHub Copilot CLI (CONNECT Proxy)

Copilot CLI, Claude Code, and Antigravity can be routed through Isartor's HTTP CONNECT
proxy so Isartor can preserve each client's native upstream as Layer 3 while still
deflecting requests locally at L1/L2 when possible:

1. A local CA certificate is generated at `~/.isartor/ca/isartor-ca.pem`
2. The CONNECT proxy runs on `:8081` (configurable via `ISARTOR__PROXY_PORT`)
3. `NODE_EXTRA_CA_CERTS` tells Copilot’s Node.js runtime to trust the local CA

```bash
# Step 1: Start Isartor (runs both API gateway :8080 and CONNECT proxy :8081)
isartor

# Step 2: Configure your client
isartor connect copilot

# Step 3: Stop any already-running client session, then activate the proxy env in the same shell
source ~/.isartor/env/copilot.sh

# Step 4: Launch the client from that same shell
# e.g. gh copilot suggest "explain this function"
```

If you start Copilot CLI, Claude Code, or Antigravity from a different shell or
from an already-running process that did not inherit the generated env vars,
traffic will bypass Isartor and you will not see proxy entries in
`isartor connect status` or `/debug/proxy/recent`.

**How it works:**
- The proxy intercepts CONNECT requests to known Copilot, Anthropic, and Antigravity domains
- TLS is terminated with a leaf certificate signed by the local CA
- POST requests to `/v1/chat/completions` and `/v1/messages` are routed through the Deflection Stack (L1a → L1b → L2 → client upstream L3)
- All other traffic is tunnelled transparently

**Intercepted domains:**
- `copilot-proxy.githubusercontent.com`
- `api.github.com`
- `api.individual.githubcopilot.com`
- `api.business.githubcopilot.com`
- `api.enterprise.githubcopilot.com`
- `api.anthropic.com`
- `cloudcode-pa.googleapis.com`
- `daily-cloudcode-pa.googleapis.com`
- `daily-cloudcode-pa.sandbox.googleapis.com`

Notes:

- The CA is only trusted by Node.js (via `NODE_EXTRA_CA_CERTS`). No system-level trust changes are made.
- Claude Code and Antigravity integrations also export `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` for non-Node HTTPS stacks.
- Some tools support overriding the OpenAI base URL directly (preferred). Point them at `http://localhost:8080/v1`.

---

For more details, see [README.md](../README.md) and [docs/2-ARCHITECTURE.md](2-ARCHITECTURE.md).


# Isartor Quickstart

Get started with Isartor in seconds using one of the following methods:

## Path A: Docker (Easiest – Batteries Included)

The fastest way to get started. All required ML models are baked into the image.

```bash
docker run -p 3000:3000 ghcr.io/isartor-ai/isartor:latest
```

## Path B: macOS & Linux (Binary)

Install the latest release with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/isartor-ai/isartor/main/scripts/install.sh | bash
```

## Path C: Windows (Binary)

Install via PowerShell one-liner:

```powershell
irm https://raw.githubusercontent.com/isartor-ai/isartor/main/scripts/install.ps1 | iex
```

> **Note for Binary Installs:**
> Unlike Docker, the raw binary requires a `config.yaml` to locate GGUF model files on your disk. See the [Configuration Guide](2-ARCHITECTURE.md#configuration) for details.

---

## Test the Prompt Firewall

You can test your Isartor instance with:

```bash
curl -X POST http://localhost:3000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Calculate 2+2"}'
```

---

### Request 1: Complex Prompt (routes to OpenAI)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-2-2b-it",
    "messages": [
      {"role": "user", "content": "Explain the quantum Hall effect in detail, including its significance for condensed matter physics and any applications in modern technology."}
    ]
  }'
```

**Expected JSON Response (snippet):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The quantum Hall effect is a phenomenon..."
      }
    }
  ],
  "usage": { ... }
}
```

**Console Log (snippet):**
```
INFO  [slm_triage] Layer 3 fallback: OpenAI
INFO  [cache] Layer 1a miss: quantum Hall effect prompt
```

### Request 2: Same Prompt (demonstrates Layer 1a Exact Cache hit)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-2-2b-it",
    "messages": [
      {"role": "user", "content": "Explain the quantum Hall effect in detail, including its significance for condensed matter physics and any applications in modern technology."}
    ]
  }'
```

**Expected JSON Response (snippet):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The quantum Hall effect is a phenomenon..."
      }
    }
  ],
  "usage": { ... }
}
```

**Console Log (snippet):**
```
INFO  [cache] Layer 1a exact match: quantum Hall effect prompt
INFO  [slm_triage] Short-circuit: cache hit
```

---

For advanced configuration, see [README.md](../README.md) and [docs/2-ARCHITECTURE.md](2-ARCHITECTURE.md).

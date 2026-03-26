# Installation

Isartor ships as a single statically linked binary — no runtime dependencies required.

## macOS / Linux — Single Command (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.sh | sh
```

## Docker

The image ships a statically linked `isartor` binary and downloads the embedding model on first start (then reuses the on-disk hf-hub cache). No API key is needed for the cache layers.

```bash
docker run -p 8080:8080 ghcr.io/isartor-ai/isartor:latest
```

To persist the model cache across restarts (recommended):

```bash
docker run -p 8080:8080 \
  -e HF_HOME=/tmp/huggingface \
  -v isartor-hf:/tmp/huggingface \
  ghcr.io/isartor-ai/isartor:latest
```

To use **Azure OpenAI** for Layer 3 (recommended: Docker secrets via `*_FILE`). Important: `ISARTOR__EXTERNAL_LLM_URL` must be the **base Azure endpoint only** (no `/openai/...` path), e.g. `https://<resource>.openai.azure.com`:

```bash
# Put your key in a file (no trailing newline is ideal, but Isartor trims whitespace)
echo -n "YOUR_AZURE_OPENAI_KEY" > ./azure_openai_key

docker run -p 8080:8080 \
  -e ISARTOR__LLM_PROVIDER=azure \
  -e ISARTOR__EXTERNAL_LLM_URL=https://<resource>.openai.azure.com \
  -e ISARTOR__AZURE_DEPLOYMENT_ID=<deployment> \
  -e ISARTOR__AZURE_API_VERSION=2024-08-01-preview \
  -e ISARTOR__EXTERNAL_LLM_API_KEY_FILE=/run/secrets/azure_openai_key \
  -v $(pwd)/azure_openai_key:/run/secrets/azure_openai_key:ro \
  ghcr.io/isartor-ai/isartor:latest
```

The startup banner appears after all layers are ready (< 30 s on a modern machine).

> **Image size:** ~120 MB compressed / ~260 MB on disk (includes `all-MiniLM-L6-v2` embedding model, statically linked Rust binary).

## Windows (PowerShell) — Single Command

```powershell
irm https://raw.githubusercontent.com/isartor-ai/Isartor/main/install.ps1 | iex
```

## Build from Source

```bash
git clone https://github.com/isartor-ai/Isartor.git
cd Isartor
cargo build --release
./target/release/isartor up
```

Requires Rust 1.75 or later.

## Verify Installation

Check that the binary is available:

```bash
isartor --version
```

Run the built-in demo. It works without an API key, but if you configure a provider first it also shows a live upstream round-trip:

```bash
isartor set-key -p groq
isartor check
isartor demo
```

Verify the health endpoint:

```bash
curl http://localhost:8080/health
# {"status":"ok","version":"0.1.0","layers":{...},"uptime_seconds":5,"demo_mode":true}
```

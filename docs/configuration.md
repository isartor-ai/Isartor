# 📄 Configuration Reference

> **Complete reference for all `ISARTOR__*` environment variables, config file support, and per-tier defaults.**

---

## Configuration Loading Order

Isartor uses the [`config`](https://crates.io/crates/config) crate with the following precedence (highest wins):

1. **Environment variables** — prefixed with `ISARTOR`, using `__` (double-underscore) as separator. Nested structs add another `__` level.
2. **Config file** — `isartor.toml`, `isartor.yaml`, or `isartor.json` in the working directory (optional).
3. **Compiled defaults** — sensible values baked into the binary.

```bash
# Example: flat field
# AppConfig.cache_backend → ISARTOR__CACHE_BACKEND
export ISARTOR__CACHE_BACKEND="redis"

# Example: nested struct field
# AppConfig.layer2.sidecar_url → ISARTOR__LAYER2__SIDECAR_URL
export ISARTOR__LAYER2__SIDECAR_URL="http://127.0.0.1:8081"
```

---

## Full Variable Reference

### General

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__HOST_PORT` | `String` | `0.0.0.0:8080` | Socket address the server binds to |
| `ISARTOR__GATEWAY_API_KEY` | `String` | `""` | API key for `X-API-Key` header (Layer 0). Empty = auth disabled (local-first). Set a value to enable gateway auth |
| `ISARTOR__INFERENCE_ENGINE` | `String` | `sidecar` | `sidecar` — uses external sidecar for Layer 2; `embedded` — uses in-process Candle engine (requires `embedded-inference` feature) |

### Layer 1 — Cache

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__CACHE_MODE` | `String` | `both` | `exact` — SHA-256 hash only; `semantic` — cosine similarity; `both` — exact first, then semantic |
| `ISARTOR__CACHE_BACKEND` | `String` | `memory` | `memory` — in-process LRU (ahash + parking_lot); `redis` — distributed Redis cache for multi-replica scaling |
| `ISARTOR__REDIS_URL` | `String` | `redis://127.0.0.1:6379` | Redis connection URI. Only used when `cache_backend=redis` |
| `ISARTOR__ROUTER_BACKEND` | `String` | `embedded` | `embedded` — in-process Candle GGUF inference; `vllm` — remote vLLM / TGI server |
| `ISARTOR__VLLM_URL` | `String` | `http://127.0.0.1:8000` | Base URL of the vLLM / TGI server. Only used when `router_backend=vllm` |
| `ISARTOR__VLLM_MODEL` | `String` | `gemma-2-2b-it` | Model name for the vLLM server. Only used when `router_backend=vllm` |
| `ISARTOR__EMBEDDING_MODEL` | `String` | `all-minilm` | Embedding model name (informational; v1 pipeline uses in-process candle BertModel) |
| `ISARTOR__SIMILARITY_THRESHOLD` | `f64` | `0.85` | Cosine similarity threshold for semantic cache hits (0.0–1.0) |
| `ISARTOR__CACHE_TTL_SECS` | `u64` | `300` | Time-to-live for cached responses, in seconds |
| `ISARTOR__CACHE_MAX_CAPACITY` | `u64` | `10000` | Maximum entries per cache (exact + semantic counted separately) |

> **Session-aware cache keys:** When clients send `x-isartor-session-id`, `x-thread-id`, `x-session-id`, or `x-conversation-id`, Isartor folds a hashed session scope into the L1 cache key and semantic-cache filter. If no session identifier is present, the previous global cache behavior is preserved. Request bodies may also supply `session_id`, `thread_id`, or `conversation_id` at the top level or under `metadata`.

> **Scalability note:** When running multiple gateway replicas (Level 3 / K8s), set `ISARTOR__CACHE_BACKEND=redis` so all pods share the same exact-match cache. With `memory` (the default), each pod maintains an independent cache, leading to lower hit rates and duplicated work.

### Layer 2 — Generation Sidecar (llama.cpp)

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__LAYER2__SIDECAR_URL` | `String` | `http://127.0.0.1:8081` | Base URL of the generation sidecar (OpenAI-compatible API) |
| `ISARTOR__LAYER2__MODEL_NAME` | `String` | `phi-3-mini` | Model name sent in the API `model` field |
| `ISARTOR__LAYER2__TIMEOUT_SECONDS` | `u64` | `30` | HTTP request timeout for sidecar calls |

### Layer 2 — Legacy (v1 Middleware, Ollama-Compatible)

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__LOCAL_SLM_URL` | `String` | `http://localhost:11434/api/generate` | URL of the local SLM for v1 middleware triage |
| `ISARTOR__LOCAL_SLM_MODEL` | `String` | `llama3` | Model name for v1 middleware requests |

### Embedding Sidecar

> **Note:** The v1 middleware pipeline (`/api/chat`) uses **in-process candle BertModel** (sentence-transformers/all-MiniLM-L6-v2) for Layer 1 embeddings — no sidecar required. These variables are only used if a separate embedding sidecar is deployed.

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__EMBEDDING_SIDECAR__SIDECAR_URL` | `String` | `http://127.0.0.1:8082` | Base URL of the embedding sidecar (`/v1/embeddings`) |
| `ISARTOR__EMBEDDING_SIDECAR__MODEL_NAME` | `String` | `all-minilm` | Embedding model name |
| `ISARTOR__EMBEDDING_SIDECAR__TIMEOUT_SECONDS` | `u64` | `10` | HTTP request timeout for embedding calls |

### Layer 3 — External Cloud LLM

All providers are powered by [rig-core](https://crates.io/crates/rig-core).

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__LLM_PROVIDER` | `String` | `openai` | Provider: `openai`, `azure`, `anthropic`, `copilot`, `xai`, `gemini`, `mistral`, `groq`, `deepseek`, `cohere`, `galadriel`, `hyperbolic`, `huggingface`, `mira`, `moonshot`, `ollama`, `openrouter`, `perplexity`, `together` |
| `ISARTOR__EXTERNAL_LLM_URL` | `String` | `https://api.openai.com/v1/chat/completions` | Base URL for the external LLM API (currently only used for the `azure` provider; other providers use their built-in endpoints) |
| `ISARTOR__EXTERNAL_LLM_MODEL` | `String` | `gpt-4o-mini` | Model name to request |
| `ISARTOR__EXTERNAL_LLM_API_KEY` | `String` | *(empty)* | API key for the cloud LLM provider (not needed for Ollama) |
| `ISARTOR__L3_TIMEOUT_SECS` | `u64` | `120` | HTTP request timeout applied to all Layer 3 provider calls, including Copilot and rig-core providers |

### Azure OpenAI (Layer 3)

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__AZURE_DEPLOYMENT_ID` | `String` | *(empty)* | Azure OpenAI deployment ID (only when `llm_provider=azure`) |
| `ISARTOR__AZURE_API_VERSION` | `String` | `2024-08-01-preview` | Azure OpenAI API version |

### Observability

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__ENABLE_MONITORING` | `bool` | `false` | Enable OpenTelemetry trace and metric export |
| `ISARTOR__OTEL_EXPORTER_ENDPOINT` | `String` | `http://localhost:4317` | OTel Collector gRPC endpoint |

### CONNECT Proxy

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ISARTOR__PROXY_PORT` | `String` | `0.0.0.0:8081` | Socket address the CONNECT proxy binds to when started via `isartor up <client>` |


### Embedded Classifier (Compiled Defaults)

These are set in `EmbeddedClassifierConfig::default()` in `src/services/local_inference.rs`. They are not currently configurable via env vars — override by modifying the source.

| Setting | Default | Description |
| --- | --- | --- |
| `repo_id` | `mradermacher/gemma-2-2b-it-GGUF` | Hugging Face repository for the GGUF model |
| `gguf_filename` | `gemma-2-2b-it.Q4_K_M.gguf` | Model file (~1.5 GB download) |
| `max_classify_tokens` | `20` | Maximum tokens for classification output |
| `max_generate_tokens` | `256` | Maximum tokens for simple task execution |
| `temperature` | `0.0` | Sampling temperature (0.0 = greedy, deterministic) |
| `repetition_penalty` | `1.1` | Repetition penalty factor |

### Rust Logging

| Variable | Default | Description |
| --- | --- | --- |
| `RUST_LOG` | *(unset)* | Standard `env_logger` / `tracing` filter (e.g., `isartor=info`, `isartor=debug,tower_http=trace`) |

---

## TOML Config File Example

Place as `isartor.toml` in the working directory:

```toml
# isartor.toml — Level 2 sidecar configuration

host_port = "0.0.0.0:8080"
gateway_api_key = "my-production-key"
inference_engine = "sidecar"

# Layer 1 — Cache
cache_mode = "both"
cache_backend = "memory"          # "memory" or "redis"
redis_url = "redis://127.0.0.1:6379"  # only used when cache_backend = "redis"
embedding_model = "all-minilm"
similarity_threshold = 0.85
cache_ttl_secs = 600
cache_max_capacity = 50000

# Layer 2 — Router backend
router_backend = "embedded"       # "embedded" or "vllm"
vllm_url = "http://127.0.0.1:8000"    # only used when router_backend = "vllm"
vllm_model = "gemma-2-2b-it"          # only used when router_backend = "vllm"

# Layer 2 — Generation Sidecar
[layer2]
sidecar_url = "http://127.0.0.1:8081"
model_name = "phi-3-mini"
timeout_seconds = 30

# Embedding Sidecar (v2 pipeline only — v1 uses in-process candle)
[embedding_sidecar]
sidecar_url = "http://127.0.0.1:8082"
model_name = "all-minilm"
timeout_seconds = 10

# Layer 3 — Cloud LLM
llm_provider = "openai"
external_llm_url = "https://api.openai.com/v1/chat/completions"
external_llm_model = "gpt-4o-mini"
external_llm_api_key = ""  # Prefer env var for secrets
l3_timeout_secs = 120

# Observability
enable_monitoring = true
otel_exporter_endpoint = "http://localhost:4317"
```

---

## Per-Tier Recommended Defaults

### Level 1 — Minimal (Edge / VPS)

```bash
ISARTOR__CACHE_MODE=both               # In-process candle BertModel enables semantic cache at all tiers
ISARTOR__CACHE_BACKEND=memory          # Single process — in-process LRU is ideal
ISARTOR__ROUTER_BACKEND=embedded       # In-process Candle SLM, no external dependencies
ISARTOR__CACHE_TTL_SECS=300
ISARTOR__CACHE_MAX_CAPACITY=5000       # Smaller memory footprint
ISARTOR__ENABLE_MONITORING=false       # No collector running
```

### Level 2 — Sidecar (Docker Compose)

```bash
ISARTOR__CACHE_MODE=both               # Semantic cache enabled
ISARTOR__CACHE_BACKEND=memory          # Single-host Docker Compose — in-process is fine
ISARTOR__ROUTER_BACKEND=embedded       # SLM runs in-process; sidecar handles generation only
ISARTOR__CACHE_TTL_SECS=300
ISARTOR__CACHE_MAX_CAPACITY=10000
ISARTOR__ENABLE_MONITORING=true
ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317
```

### Level 3 — Enterprise (Kubernetes)

```bash
ISARTOR__CACHE_MODE=both
ISARTOR__CACHE_BACKEND=redis           # ← Shared cache across all pods for consistent hit rates
ISARTOR__REDIS_URL=redis://redis.isartor:6379
ISARTOR__ROUTER_BACKEND=vllm           # ← GPU-backed vLLM server for high-throughput SLM routing
ISARTOR__VLLM_URL=http://vllm.isartor:8000
ISARTOR__VLLM_MODEL=gemma-2-2b-it
ISARTOR__CACHE_TTL_SECS=600            # Longer TTL, more cache value
ISARTOR__CACHE_MAX_CAPACITY=100000     # Large cache for high traffic
ISARTOR__ENABLE_MONITORING=true
ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector.isartor:4317
```

> **Scalability:** With `cache_backend=redis`, every gateway replica shares the same exact-match cache, ensuring cache hits are consistent regardless of which pod serves the request. With `router_backend=vllm`, the SLM classification workload is offloaded to a GPU-backed vLLM cluster that can be independently scaled (more GPU replicas = higher throughput). This allows the stateless Isartor gateway pods to scale horizontally to hundreds of replicas without contention.

---

## Provider-Specific Configuration

### OpenAI

```bash
ISARTOR__LLM_PROVIDER=openai
ISARTOR__EXTERNAL_LLM_URL=https://api.openai.com/v1/chat/completions
ISARTOR__EXTERNAL_LLM_MODEL=gpt-4o-mini
ISARTOR__EXTERNAL_LLM_API_KEY=sk-...
```

### Azure OpenAI

```bash
ISARTOR__LLM_PROVIDER=azure
ISARTOR__EXTERNAL_LLM_URL=https://your-resource.openai.azure.com
ISARTOR__EXTERNAL_LLM_MODEL=gpt-4o-mini
ISARTOR__EXTERNAL_LLM_API_KEY=your-azure-key
ISARTOR__AZURE_DEPLOYMENT_ID=your-deployment-name
ISARTOR__AZURE_API_VERSION=2024-08-01-preview
```

### Anthropic

```bash
ISARTOR__LLM_PROVIDER=anthropic
ISARTOR__EXTERNAL_LLM_URL=https://api.anthropic.com/v1/messages
ISARTOR__EXTERNAL_LLM_MODEL=claude-3-5-sonnet-20241022
ISARTOR__EXTERNAL_LLM_API_KEY=sk-ant-...
```

### xAI / Grok

```bash
ISARTOR__LLM_PROVIDER=xai
ISARTOR__EXTERNAL_LLM_URL=https://api.x.ai/v1/chat/completions
ISARTOR__EXTERNAL_LLM_MODEL=grok-2
ISARTOR__EXTERNAL_LLM_API_KEY=xai-...
```

### Google Gemini

```bash
ISARTOR__LLM_PROVIDER=gemini
ISARTOR__EXTERNAL_LLM_MODEL=gemini-2.0-flash
ISARTOR__EXTERNAL_LLM_API_KEY=AIza...
```

### Mistral AI

```bash
ISARTOR__LLM_PROVIDER=mistral
ISARTOR__EXTERNAL_LLM_MODEL=mistral-small-latest
ISARTOR__EXTERNAL_LLM_API_KEY=...
```

### Groq (Llama, Mixtral)

```bash
ISARTOR__LLM_PROVIDER=groq
ISARTOR__EXTERNAL_LLM_MODEL=llama-3.1-8b-instant
ISARTOR__EXTERNAL_LLM_API_KEY=gsk_...
```

### DeepSeek

```bash
ISARTOR__LLM_PROVIDER=deepseek
ISARTOR__EXTERNAL_LLM_MODEL=deepseek-chat
ISARTOR__EXTERNAL_LLM_API_KEY=sk-...
```

### Cohere

```bash
ISARTOR__LLM_PROVIDER=cohere
ISARTOR__EXTERNAL_LLM_MODEL=command-r
ISARTOR__EXTERNAL_LLM_API_KEY=...
```

### Galadriel

```bash
ISARTOR__LLM_PROVIDER=galadriel
ISARTOR__EXTERNAL_LLM_MODEL=gpt-4o
ISARTOR__EXTERNAL_LLM_API_KEY=...
```

### Hyperbolic

```bash
ISARTOR__LLM_PROVIDER=hyperbolic
ISARTOR__EXTERNAL_LLM_MODEL=deepseek-ai/DeepSeek-R1
ISARTOR__EXTERNAL_LLM_API_KEY=...
```

### HuggingFace Inference API

```bash
ISARTOR__LLM_PROVIDER=huggingface
ISARTOR__EXTERNAL_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
ISARTOR__EXTERNAL_LLM_API_KEY=hf_...
```

### Mira

```bash
ISARTOR__LLM_PROVIDER=mira
ISARTOR__EXTERNAL_LLM_MODEL=gpt-4o
ISARTOR__EXTERNAL_LLM_API_KEY=...
```

### Moonshot (Kimi)

```bash
ISARTOR__LLM_PROVIDER=moonshot
ISARTOR__EXTERNAL_LLM_MODEL=moonshot-v1-128k
ISARTOR__EXTERNAL_LLM_API_KEY=...
```

### Ollama (local — no API key required)

```bash
ISARTOR__LLM_PROVIDER=ollama
ISARTOR__EXTERNAL_LLM_MODEL=llama3.2
# No API key needed — Ollama runs locally on http://localhost:11434
```

### OpenRouter

```bash
ISARTOR__LLM_PROVIDER=openrouter
ISARTOR__EXTERNAL_LLM_MODEL=openai/gpt-4o-mini
ISARTOR__EXTERNAL_LLM_API_KEY=sk-or-...
```

### Perplexity

```bash
ISARTOR__LLM_PROVIDER=perplexity
ISARTOR__EXTERNAL_LLM_MODEL=sonar
ISARTOR__EXTERNAL_LLM_API_KEY=pplx-...
```

### Together AI

```bash
ISARTOR__LLM_PROVIDER=together
ISARTOR__EXTERNAL_LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
ISARTOR__EXTERNAL_LLM_API_KEY=...
```

---

## Setting Provider Keys via CLI

The `isartor set-key` command lets you configure LLM provider API keys, model names, and provider selection directly from the command line. It writes to `isartor.toml` by default (preserving existing settings) or to a shell-sourceable env file.

### Usage

```bash
# Interactive — prompts for the key securely (hidden input)
isartor set-key --provider openai

# Inline (for scripts / CI)
isartor set-key --provider anthropic --key sk-ant-xxxxx --model claude-sonnet-4-6

# Groq with default model (llama-3.1-8b-instant)
isartor set-key --provider groq --key gsk_xxxxx

# Write to env file instead of TOML
isartor set-key --provider deepseek --key sk-xxxxx --env-file

# Dry-run — see what would change without writing
isartor set-key --provider gemini --key AIza... --dry-run

# Ollama (no key needed)
isartor set-key --provider ollama

# Custom config path
isartor set-key --provider openai --key sk-... --config-path /etc/isartor/isartor.toml
```

### Arguments

| Argument | Short | Required | Description |
|---|---|---|---|
| `--provider` | `-p` | Yes | LLM provider name (e.g., `openai`, `anthropic`, `groq`, `ollama`) |
| `--key` | `-k` | No | API key. If omitted and provider requires one, prompts interactively |
| `--model` | `-m` | No | Model name. Uses sensible default for the provider if omitted |
| `--config-path` | | No | Path to `isartor.toml` (default: `./isartor.toml`) |
| `--dry-run` | | No | Print what would be written without modifying files |
| `--env-file` | | No | Write `export` statements to `~/.isartor/env` instead of TOML |

### Default Models

When `--model` is omitted, the following defaults are used:

| Provider | Default Model |
|---|---|
| `openai` | `gpt-4o-mini` |
| `azure` | `gpt-4o-mini` |
| `anthropic` | `claude-3-5-sonnet-20241022` |
| `xai` | `grok-2` |
| `gemini` | `gemini-2.0-flash` |
| `mistral` | `mistral-small-latest` |
| `groq` | `llama-3.1-8b-instant` |
| `deepseek` | `deepseek-chat` |
| `cohere` | `command-r` |
| `ollama` | `llama3.2` |
| `openrouter` | `openai/gpt-4o-mini` |
| `perplexity` | `sonar` |
| `together` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| All others | `gpt-4o-mini` |

---

*← Back to [README](../README.md)*

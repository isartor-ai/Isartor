# Configuration Reference

> **Complete reference for every Isartor configuration variable, CLI command, and provider option.**

---

## Configuration Loading Order

Isartor loads configuration in the following order (later sources override earlier ones):

1. **Compiled defaults** — baked into the binary
2. **`isartor.toml`** — if present in the working directory or `~/.isartor/`
3. **Environment variables** — `ISARTOR__...` with double-underscore separators

Generate a starter config file with:

```bash
isartor init
```

---

## Master Configuration Table

| YAML Key                 | Environment Variable            | Type     | Default                | Description                                      |
|--------------------------|-------------------------------- |----------|------------------------|--------------------------------------------------|
| server.host              | ISARTOR__HOST                   | string   | 0.0.0.0                | Host address for server binding                  |
| server.port              | ISARTOR__PORT                   | int      | 8080                   | Port for HTTP server                             |
| exact_cache.provider     | ISARTOR__CACHE_BACKEND          | string   | memory                 | Layer 1a cache backend: memory or redis          |
| exact_cache.redis_url    | ISARTOR__REDIS_URL              | string   | (none)                 | Redis connection string (if provider=redis)      |
| exact_cache.redis_db     | ISARTOR__REDIS_DB               | int      | 0                      | Redis database index                             |
| semantic_cache.provider  | ISARTOR__SEMANTIC_BACKEND       | string   | candle                 | Layer 1b semantic cache: candle (in-process) or tei (external) |
| semantic_cache.remote_url| ISARTOR__TEI_URL                | string   | (none)                 | TEI endpoint (if provider=tei)                   |
| slm_router.provider      | ISARTOR__ROUTER_BACKEND         | string   | embedded               | Layer 2 router: embedded or vllm                 |
| slm_router.remote_url    | ISARTOR__VLLM_URL               | string   | (none)                 | vLLM/TGI endpoint (if provider=vllm)             |
| slm_router.model         | ISARTOR__VLLM_MODEL             | string   | gemma-2-2b-it          | Model name/path for SLM router                   |
| slm_router.model_path    | ISARTOR__MODEL_PATH             | string   | (baked-in)             | Path to GGUF model file (embedded mode)          |
| slm_router.classifier_mode | ISARTOR__LAYER2__CLASSIFIER_MODE | string | tiered               | Classifier mode: `tiered` (TEMPLATE/SNIPPET/COMPLEX) or `binary` (legacy SIMPLE/COMPLEX) |
| slm_router.max_answer_tokens | ISARTOR__LAYER2__MAX_ANSWER_TOKENS | u64 | 2048                | Max tokens the SLM may generate for a local answer |
| fallback.openai_api_key  | ISARTOR__OPENAI_API_KEY         | string   | (none)                 | OpenAI API key for Layer 3 fallback              |
| fallback.anthropic_api_key| ISARTOR__ANTHROPIC_API_KEY     | string   | (none)                 | Anthropic API key for Layer 3 fallback           |
| llm_provider             | ISARTOR__LLM_PROVIDER           | string   | openai                 | LLM provider (see below for full list)           |
| external_llm_model       | ISARTOR__EXTERNAL_LLM_MODEL     | string   | gpt-4o-mini            | Model name to request from the provider          |
| external_llm_api_key     | ISARTOR__EXTERNAL_LLM_API_KEY   | string   | (none)                 | API key for the configured LLM provider (not needed for ollama) |
| l3_timeout_secs          | ISARTOR__L3_TIMEOUT_SECS        | u64      | 120                    | HTTP timeout applied to all Layer 3 provider requests |
| enable_context_optimizer | ISARTOR__ENABLE_CONTEXT_OPTIMIZER | bool   | true                   | Master switch for L2.5 context optimiser             |
| context_optimizer_dedup  | ISARTOR__CONTEXT_OPTIMIZER_DEDUP | bool    | true                   | Enable cross-turn instruction deduplication          |
| context_optimizer_minify | ISARTOR__CONTEXT_OPTIMIZER_MINIFY | bool   | true                   | Enable static minification (comments, rules, blanks) |

---

## Sections

### Server

- `server.host`, `server.port`: Bind address and port.

### Layer 1a: Exact Match Cache

- `exact_cache.provider`: `memory` or `redis`
- `exact_cache.redis_url`, `exact_cache.redis_db`: Redis config

### Layer 1b: Semantic Cache

- `semantic_cache.provider`: `candle` or `tei`
- `semantic_cache.remote_url`: TEI endpoint
- Requests that carry `x-isartor-session-id`, `x-thread-id`, `x-session-id`, or
  `x-conversation-id` are isolated into a session-aware cache scope. The same
  scope can also be provided in request bodies via `session_id`, `thread_id`,
  `conversation_id`, or `metadata.*`. If no session identifier is present,
  Isartor keeps the legacy global-cache behavior.

### Layer 2: SLM Router

- `slm_router.provider`: `embedded` or `vllm`
- `slm_router.remote_url`, `slm_router.model`, `slm_router.model_path`: Router config
- `slm_router.classifier_mode`: `tiered` (default — TEMPLATE/SNIPPET/COMPLEX) or `binary` (legacy SIMPLE/COMPLEX)
- `slm_router.max_answer_tokens`: Max tokens the SLM may generate for a local answer (default 2048)

### Layer 2.5: Context Optimiser

L2.5 compresses repeated instruction payloads (CLAUDE.md, copilot-instructions.md, skills blocks) before they reach the cloud, reducing input tokens on every L3 call.

- `enable_context_optimizer`: Master switch (default `true`). Set to `false` to disable L2.5 entirely.
- `context_optimizer_dedup`: Enable cross-turn instruction deduplication (default `true`). When the same instruction block is seen in consecutive turns of the same session, it is replaced with a compact hash reference.
- `context_optimizer_minify`: Enable static minification (default `true`). Strips HTML/XML comments, decorative horizontal rules, consecutive blank lines, and Unicode box-drawing decoration.

The pipeline processes system/instruction messages from OpenAI, Anthropic, and native request formats. See [Deflection Stack — L2.5](../concepts/deflection-stack.md#l25--context-optimiser) for architecture details.

### Layer 3: Cloud Fallbacks

- `fallback.openai_api_key`, `fallback.anthropic_api_key`: API keys for external LLMs
- `llm_provider`: Select the active provider. All providers are powered by [rig-core](https://crates.io/crates/rig-core) except `copilot`, which uses Isartor's native GitHub Copilot adapter:
  - `openai` (default), `azure`, `anthropic`, `xai`
  - `gemini`, `mistral`, `groq`, `deepseek`
  - `cohere`, `galadriel`, `hyperbolic`, `huggingface`
  - `mira`, `moonshot`, `ollama` (local, no key), `openrouter`
  - `perplexity`, `together`
  - `copilot` (GitHub Copilot subscription-backed L3)
- `external_llm_model`: Model name for the selected provider (e.g. `gpt-4o-mini`, `gemini-2.0-flash`, `mistral-small-latest`, `llama-3.1-8b-instant`, `deepseek-chat`, `command-r`, `sonar`, `moonshot-v1-128k`)
- `external_llm_api_key`: API key for the configured provider (not needed for `ollama`)
- `l3_timeout_secs`: Shared timeout, in seconds, for all Layer 3 provider HTTP calls

---

## TOML Config Example

Generate a scaffold with `isartor init`, then edit `isartor.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080

[exact_cache]
provider = "memory"           # "memory" or "redis"
# redis_url = "redis://127.0.0.1:6379"
# redis_db = 0

[semantic_cache]
provider = "candle"           # "candle" or "tei"
# remote_url = "http://localhost:8082"

[slm_router]
provider = "embedded"         # "embedded" or "vllm"
# remote_url = "http://localhost:8000"
# model = "gemma-2-2b-it"

# L2.5 Context Optimiser (all enabled by default)
# enable_context_optimizer = true
# context_optimizer_dedup = true
# context_optimizer_minify = true

[fallback]
# openai_api_key = "sk-..."
# anthropic_api_key = "sk-ant-..."

# llm_provider = "openai"
# external_llm_model = "gpt-4o-mini"
# external_llm_api_key = "sk-..."
```

---

## Per-Tier Defaults

| Setting | Level 1 (Minimal) | Level 2 (Sidecar) | Level 3 (Enterprise) |
|---------|--------------------|--------------------|----------------------|
| Cache backend | memory | memory | redis |
| Semantic backend | candle | candle | tei (optional) |
| SLM router | embedded | embedded or sidecar | vllm |
| LLM provider | openai | openai | any |
| Monitoring | false | true | true |

---

## Provider-Specific Configuration

Each provider requires `ISARTOR__EXTERNAL_LLM_API_KEY` (except Ollama) and a matching `ISARTOR__LLM_PROVIDER` value:

```bash
# OpenAI (default)
export ISARTOR__LLM_PROVIDER=openai
export ISARTOR__EXTERNAL_LLM_MODEL=gpt-4o-mini

# Azure OpenAI
export ISARTOR__LLM_PROVIDER=azure

# Anthropic
export ISARTOR__LLM_PROVIDER=anthropic
export ISARTOR__EXTERNAL_LLM_MODEL=claude-3-haiku-20240307

# xAI (Grok)
export ISARTOR__LLM_PROVIDER=xai

# Google Gemini
export ISARTOR__LLM_PROVIDER=gemini
export ISARTOR__EXTERNAL_LLM_MODEL=gemini-2.0-flash

# Ollama (local — no API key required)
export ISARTOR__LLM_PROVIDER=ollama
export ISARTOR__EXTERNAL_LLM_MODEL=llama3

# GitHub Copilot (configured automatically by `isartor connect claude-copilot`)
export ISARTOR__LLM_PROVIDER=copilot
export ISARTOR__EXTERNAL_LLM_MODEL=claude-sonnet-4.5
```

---

## Setting API Keys with the CLI

Use `isartor set-key` for interactive key management:

```bash
isartor set-key --provider openai
isartor set-key --provider anthropic
isartor set-key --provider xai
```

This writes the key to `isartor.toml` or the appropriate env file.

---

## CLI Commands

| Command | Description |
|---|---|
| `isartor up` | Start the API gateway only (recommended default). Flag: `--detach` to run in background |
| `isartor up <copilot\|claude\|antigravity>` | Start the gateway plus the CONNECT proxy for that client |
| `isartor init` | Generate a commented `isartor.toml` config scaffold |
| `isartor demo` | Run the post-install showcase (cache-only, or live + cache when a provider is configured) |
| `isartor check` | Audit outbound connections |
| `isartor connect <client>` | Configure AI clients to route through Isartor |
| `isartor connect copilot` | Configure Copilot CLI with CONNECT proxy + TLS MITM |
| `isartor connect claude-copilot` | Configure Claude Code to use GitHub Copilot through Isartor |
| `isartor stats` | Show total prompts, counts by layer, and recent prompt routing history |
| `isartor set-key --provider <name>` | Set LLM provider API key (writes to `isartor.toml` or env file) |
| `isartor stop` | Stop a running Isartor instance (uses PID file). Flags: `--force` (SIGKILL), `--pid-file <path>` |
| `isartor update` | Self-update to the latest (or specific) version. Flags: `--version <tag>`, `--dry-run`, `--force` |

---

*See also: [Architecture](../concepts/architecture.md) · [Metrics & Tracing](../observability/metrics-tracing.md) · [Troubleshooting](../development/troubleshooting.md)*

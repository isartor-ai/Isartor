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
| model_aliases.<name>     | ISARTOR__MODEL_ALIASES__<NAME>  | string   | (none)                 | Request-time alias that resolves to a real model ID |
| external_llm_api_key     | ISARTOR__EXTERNAL_LLM_API_KEY   | string   | (none)                 | API key for the configured LLM provider (not needed for ollama) |
| provider_keys            | ISARTOR__PROVIDER_KEYS          | JSON array | []                   | Optional multi-key pool for the primary provider |
| key_rotation_strategy    | ISARTOR__KEY_ROTATION_STRATEGY  | string   | round_robin            | Multi-key selection strategy: `round_robin` or `priority` |
| key_cooldown_secs        | ISARTOR__KEY_COOLDOWN_SECS      | u64      | 60                     | Cooldown applied after a key hits rate limits or quota exhaustion |
| l3_timeout_secs          | ISARTOR__L3_TIMEOUT_SECS        | u64      | 120                    | HTTP timeout applied to all Layer 3 provider requests |
| enable_context_optimizer | ISARTOR__ENABLE_CONTEXT_OPTIMIZER | bool   | true                   | Master switch for L2.5 context optimiser             |
| context_optimizer_dedup  | ISARTOR__CONTEXT_OPTIMIZER_DEDUP | bool    | true                   | Enable cross-turn instruction deduplication          |
| context_optimizer_minify | ISARTOR__CONTEXT_OPTIMIZER_MINIFY | bool   | true                   | Enable static minification (comments, rules, blanks) |
| enable_request_logs      | ISARTOR__ENABLE_REQUEST_LOGS     | bool    | false                  | Opt-in request/response debug logging with redaction |
| request_log_path         | ISARTOR__REQUEST_LOG_PATH        | string  | ~/.isartor/request_logs | Directory for rotating JSONL request logs            |
| usage_log_path           | ISARTOR__USAGE_LOG_PATH          | string  | ~/.isartor             | Directory that stores `usage.jsonl` for usage stats and quotas |
| usage_retention_days     | ISARTOR__USAGE_RETENTION_DAYS    | u64     | 30                     | Retention window for persisted usage events          |
| usage_window_hours       | ISARTOR__USAGE_WINDOW_HOURS      | u64     | 24                     | Default reporting window for `isartor stats --usage` |
| quota.<provider>.*       | ISARTOR__QUOTA__<PROVIDER>__*    | mixed   | (none)                 | Per-provider token/cost quota policy and action      |

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

### Request debug logging

Isartor can optionally record request and response payloads to a separate JSONL log for troubleshooting provider or client integrations.

- `enable_request_logs`: Default `false`. Set to `true` only while debugging.
- `request_log_path`: Directory where rotating request log files are written. Default `~/.isartor/request_logs`.

Important behavior:

- request logs are separate from `isartor.log` and OpenTelemetry output
- sensitive headers such as `Authorization`, `api-key`, and `x-api-key` are redacted automatically
- bodies are truncated to a bounded size per entry to keep logs manageable
- `isartor logs --requests` shows or follows the request log stream

### Layer 3: Cloud Fallbacks

- `fallback.openai_api_key`, `fallback.anthropic_api_key`: API keys for external LLMs
- `llm_provider`: Select the active provider. All providers are powered by [rig-core](https://crates.io/crates/rig-core) except `copilot`, which uses Isartor's native GitHub Copilot adapter:
  - `openai` (default), `azure`, `anthropic`, `xai`
  - `gemini`, `mistral`, `groq`, `cerebras`, `nebius`, `siliconflow`, `fireworks`, `nvidia`, `chutes`, `deepseek`
  - `cohere`, `galadriel`, `hyperbolic`, `huggingface`
  - `mira`, `moonshot`, `ollama` (local, no key), `openrouter`
  - `perplexity`, `together`
  - `copilot` (GitHub Copilot subscription-backed L3)
- `external_llm_model`: Model name for the selected provider (e.g. `gpt-4o-mini`, `gemini-2.0-flash`, `mistral-small-latest`, `llama-3.1-8b-instant`, `deepseek-chat`, `command-r`, `sonar`, `moonshot-v1-128k`)
- Many OpenAI-compatible providers ship with built-in default endpoints now, so `set-key`, `setup`, and `check` work directly for providers such as Cerebras, Nebius, SiliconFlow, Fireworks, NVIDIA, and Chutes.
- `model_aliases`: Optional map of friendly names to real model IDs. Alias resolution happens at the HTTP boundary before L1 cache keys are built, so `model="fast"` and the resolved real model share the same canonical cache behavior.
- `external_llm_api_key`: API key for the configured provider (not needed for `ollama`)
- `provider_keys`: Optional array-of-tables or `ISARTOR__PROVIDER_KEYS` JSON array for multiple credentials on the same provider. Each entry supports `key`, `priority`, and optional `label`.
- `key_rotation_strategy`: `round_robin` (default) or `priority`
- `key_cooldown_secs`: Cooldown window, in seconds, after a key hits `429` / quota-style upstream failures
- `l3_timeout_secs`: Shared timeout, in seconds, for all Layer 3 provider HTTP calls
- `fallback_providers`: Optional ordered backup chain. Keep the current top-level provider as the primary, then add `[[fallback_providers]]` entries with `provider`, `model`, `api_key`, `provider_keys`, `key_rotation_strategy`, and `url`. Azure fallbacks can also set `azure_deployment_id` and `azure_api_version`.
- `ISARTOR__FALLBACK_PROVIDERS`: Environment override for the same chain as a JSON array of provider objects. Example:

```bash
export ISARTOR__FALLBACK_PROVIDERS='[
  {"provider":"nvidia","model":"meta/llama-3.1-8b-instruct","api_key":"nvapi-...","url":"https://integrate.api.nvidia.com/v1/chat/completions"},
  {"provider":"openrouter","model":"openai/gpt-4o-mini","api_key":"sk-or-...","url":"https://openrouter.ai/api/v1/chat/completions"}
]'
```

- Failover happens only after the current provider exhausts its own retry budget, and only for provider-side errors that are safe to cascade (for example `429`, `5xx`, timeouts, and quota-style failures). Invalid request / bad request errors do **not** move to the next provider.
- Successful Layer 3 responses include an `x-isartor-provider` header naming the upstream that actually answered.

### Provider status / health

Isartor keeps an in-memory status tracker for the configured Layer 3 chain. It is intentionally process-local and resets on restart.

- `GET /debug/providers`: Authenticated debug endpoint that returns the active provider plus every configured primary/fallback entry, including model, endpoint, request/error counts, and the last-known success/error timestamps and message.
- Provider status now also includes masked key-pool entries, their strategy, request counts, rate-limit counts, and cooldown state.
- `isartor providers`: CLI view that reads `/debug/providers` when the gateway is reachable and falls back to local config inspection when it is not.
- The tracker is updated only by real Layer 3 request outcomes. It does not persist across restarts and does not write to Redis or other storage.

### Supported inbound API surfaces

Isartor currently accepts four user-facing request formats at the gateway boundary:

- Native Isartor: `POST /api/chat` and `POST /api/v1/chat`
- OpenAI-compatible: `POST /v1/chat/completions`
- Anthropic-compatible: `POST /v1/messages`
- Gemini-native: `POST /v1beta/models/{model}:generateContent` and `POST /v1beta/models/{model}:streamGenerateContent`

Gemini-native requests use the model embedded in the URL path as the canonical request model when the body omits a `model` field. Cache entries stay namespaced by API surface, so Gemini JSON responses never collide with native, OpenAI, or Anthropic cache entries even when the underlying prompt text is identical.

### Model aliases

Use aliases when you want clients to send stable names like `fast`, `smart`, or `code` instead of raw provider model IDs:

```toml
[model_aliases]
fast = "gpt-4o-mini"
smart = "gpt-4o"
code = "gpt-4.1"
```

You can also write them from the CLI:

```bash
isartor set-alias --alias fast --model gpt-4o-mini
```

Aliases are currently model aliases within the configured provider. They are surfaced by `GET /v1/models` alongside the configured real model IDs.

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
# key_rotation_strategy = "round_robin"
# key_cooldown_secs = 60
#
# [[provider_keys]]
# key = "sk-primary"
# priority = 1
# label = "primary"
#
# [[provider_keys]]
# key = "sk-team-shared"
# priority = 2
# label = "team-shared"
#
# [[fallback_providers]]
# provider = "nvidia"
# model = "meta/llama-3.1-8b-instruct"
# api_key = "nvapi-..."
# url = "https://integrate.api.nvidia.com/v1/chat/completions"
#
# [model_aliases]
# fast = "gpt-4o-mini"
# smart = "gpt-4o"
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
| `isartor set-alias --alias <name> --model <id>` | Set a request-time model alias in `isartor.toml` |
| `isartor providers` | Show the active provider config plus last-known in-memory Layer 3 health |
| `isartor logs --requests` | Show or follow the separate request/response debug log |
| `isartor stop` | Stop a running Isartor instance (uses PID file). Flags: `--force` (SIGKILL), `--pid-file <path>` |
| `isartor update` | Self-update to the latest (or specific) version. Flags: `--version <tag>`, `--dry-run`, `--force` |

---

*See also: [Architecture](../concepts/architecture.md) · [Metrics & Tracing](../observability/metrics-tracing.md) · [Troubleshooting](../development/troubleshooting.md)*


## Usage analytics

Isartor can persist provider/model usage events to `~/.isartor/usage.jsonl` and expose aggregated summaries through `GET /debug/usage` and `isartor stats --usage`. Configure `usage_log_path`, `usage_retention_days`, `usage_window_hours`, and `usage_pricing.<provider>.{input_cost_per_million_usd,output_cost_per_million_usd}` via `ISARTOR__...` environment variables or `isartor.toml`.

## Provider quotas

Quota policies reuse the same persisted usage tracker instead of introducing a second accounting store. Define a policy per provider with `[quota.<provider>]`, then set any mix of token and cost ceilings:

- `daily_token_limit`, `weekly_token_limit`, `monthly_token_limit`
- `daily_cost_limit_usd`, `weekly_cost_limit_usd`, `monthly_cost_limit_usd`
- `warning_threshold_ratio` (default `0.8`)
- `action_on_limit` = `warn`, `block`, or `fallback`

Behavior notes:

- warnings are emitted when projected usage for the in-flight request crosses the configured threshold
- `block` returns HTTP `429`
- `fallback` skips the current provider and continues down the existing Layer 3 provider chain
- quota windows reset on UTC boundaries: daily at midnight, weekly on Monday 00:00, monthly on the first day of the month
- `isartor check` prints the current quota window status for each configured provider target

Example:

```toml
[quota.openai]
daily_token_limit = 500000
monthly_cost_limit_usd = 25.0
warning_threshold_ratio = 0.8
action_on_limit = "fallback"

[quota.anthropic]
daily_cost_limit_usd = 10.0
action_on_limit = "block"
```

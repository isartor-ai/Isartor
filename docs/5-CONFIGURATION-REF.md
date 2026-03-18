# Isartor Configuration Reference

## Master Configuration Table

| YAML Key                | Environment Variable           | Type     | Default                | Description                                      |
|------------------------ |------------------------------- |----------|------------------------|--------------------------------------------------|
| server.host             | ISARTOR__HOST                  | string   | 0.0.0.0                | Host address for server binding                  |
| server.port             | ISARTOR__PORT                  | int      | 8080                   | Port for HTTP server                             |
| exact_cache.provider    | ISARTOR__CACHE_BACKEND         | string   | memory                 | Layer 1a cache backend: memory or redis          |
| exact_cache.redis_url   | ISARTOR__REDIS_URL             | string   | (none)                 | Redis connection string (if provider=redis)      |
| exact_cache.redis_db    | ISARTOR__REDIS_DB              | int      | 0                      | Redis database index                             |
| semantic_cache.provider | ISARTOR__SEMANTIC_BACKEND      | string   | candle                 | Layer 1b semantic cache: candle (in-process) or tei (external) |
| semantic_cache.remote_url| ISARTOR__TEI_URL               | string   | (none)                 | TEI endpoint (if provider=tei)                   |
| slm_router.provider     | ISARTOR__ROUTER_BACKEND        | string   | embedded               | Layer 2 router: embedded or vllm                 |
| slm_router.remote_url   | ISARTOR__VLLM_URL              | string   | (none)                 | vLLM/TGI endpoint (if provider=vllm)             |
| slm_router.model        | ISARTOR__VLLM_MODEL            | string   | gemma-2-2b-it          | Model name/path for SLM router                   |
| slm_router.model_path   | ISARTOR__MODEL_PATH            | string   | (baked-in)             | Path to GGUF model file (embedded mode)          |
| fallback.openai_api_key | ISARTOR__OPENAI_API_KEY        | string   | (none)                 | OpenAI API key for Layer 3 fallback              |
| fallback.anthropic_api_key| ISARTOR__ANTHROPIC_API_KEY   | string   | (none)                 | Anthropic API key for Layer 3 fallback           |
| llm_provider            | ISARTOR__LLM_PROVIDER          | string   | openai                 | LLM provider (see below for full list)           |
| external_llm_model      | ISARTOR__EXTERNAL_LLM_MODEL    | string   | gpt-4o-mini            | Model name to request from the provider          |
| external_llm_api_key    | ISARTOR__EXTERNAL_LLM_API_KEY  | string   | (none)                 | API key for the configured LLM provider (not needed for ollama) |

## Sections

### Server
- `server.host`, `server.port`: Bind address and port.

### Layer 1a: Exact Match Cache
- `exact_cache.provider`: `memory` or `redis`
- `exact_cache.redis_url`, `exact_cache.redis_db`: Redis config

### Layer 1b: Semantic Cache
- `semantic_cache.provider`: `candle` or `tei`
- `semantic_cache.remote_url`: TEI endpoint

### Layer 2: SLM Router
- `slm_router.provider`: `embedded` or `vllm`
- `slm_router.remote_url`, `slm_router.model`, `slm_router.model_path`: Router config

### Layer 3: Cloud Fallbacks
- `fallback.openai_api_key`, `fallback.anthropic_api_key`: API keys for external LLMs
- `llm_provider`: Select the active provider. All providers are powered by [rig-core](https://crates.io/crates/rig-core):
  - `openai` (default), `azure`, `anthropic`, `xai`
  - `gemini`, `mistral`, `groq`, `deepseek`
  - `cohere`, `galadriel`, `hyperbolic`, `huggingface`
  - `mira`, `moonshot`, `ollama` (local, no key), `openrouter`
  - `perplexity`, `together`
- `external_llm_model`: Model name for the selected provider (e.g. `gpt-4o-mini`, `gemini-2.0-flash`, `mistral-small-latest`, `llama-3.1-8b-instant`, `deepseek-chat`, `command-r`, `sonar`, `moonshot-v1-128k`)
- `external_llm_api_key`: API key for the configured provider (not needed for `ollama`)

### CLI Commands

| Command | Description |
|---|---|
| `isartor` | Start the server (default) |
| `isartor init` | Generate a commented `isartor.toml` config scaffold |
| `isartor demo` | Run the deflection demo (no API key needed) |
| `isartor connectivity-check` | Audit outbound connections |
| `isartor connect <client>` | Configure AI clients to route through Isartor |
| `isartor connect copilot` | Configure Copilot CLI with CONNECT proxy + TLS MITM |
| `isartor set-key --provider <name>` | Set LLM provider API key (writes to `isartor.toml` or env file) |
| `isartor stop` | Stop a running Isartor instance (uses PID file). Flags: `--force` (SIGKILL), `--pid-file <path>` |
| `isartor update` | Self-update to the latest (or specific) version from GitHub releases. Flags: `--version <tag>`, `--dry-run`, `--force` |

---

For full details, see [README.md](../README.md) and [docs/2-ARCHITECTURE.md](2-ARCHITECTURE.md).

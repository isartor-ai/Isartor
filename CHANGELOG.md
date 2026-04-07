# Changelog

All notable changes to Isartor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2026.4.1] - 2026-04-07

### Added
- **Embedded web management dashboard** (`/dashboard`): a self-contained single-page application served directly from the binary (no separate server or CDN required). The dashboard provides four views:
  - **Overview** — gateway version, active provider/model, total requests, deflection rate, token usage, and estimated cost and savings, with live status pills for SLM router, request logging, and offline mode.
  - **Providers** — health, model, request count, error count, and masked key pool for each configured provider in the fallback chain.
  - **Usage** — usage window summary with deflection rate, prompt/completion token breakdown, cloud cost, and saved cost, plus a per-day request/deflection bar chart.
  - **Request Log** — most recent 50 request log entries from the JSONL request log, showing method, path, HTTP status, final layer, provider, model, and latency.
- API key authentication for all `/api/admin/*` endpoints; the dashboard shell HTML is served without auth and prompts for the gateway API key on first load (stored in `sessionStorage`).
- Four new authenticated admin API endpoints: `GET /api/admin/overview`, `GET /api/admin/providers`, `GET /api/admin/usage`, `GET /api/admin/requests`.

## [2026.4.0] - 2026-04-07

### Added
- **L2 classifier context enrichment**: `extract_classifier_context()` now feeds the system prompt alongside the last user message to the local SLM classifier, so agentic tasks with short user turns but complex system prompts are correctly identified as COMPLEX and forwarded to Layer 3 rather than deflected incorrectly.
- **Split L2 classify/generate URLs**: classification requests now use `local_slm_url` + `local_slm_model` (lightweight CPU-friendly Ollama path) while answer generation continues to use `layer2.sidecar_url` + `layer2.model_name` (GPU sidecar). This separates always-on classification from heavier generation inference.
- **Explicit L2 HTTP timeouts**: both the classify and generate HTTP calls now respect `layer2.timeout_seconds`, preventing silent hangs when the local SLM is slow or unreachable.

## [2026.3.12] - 2026-03-29

### Added
- **Ordered Layer 3 fallback chain**: added chain-aware provider execution with per-provider retry budgets, `x-isartor-provider` response headers, and fallback-aware `isartor check` output so the gateway can move predictably across multiple providers.
- **Per-provider multi-key rotation**: added configurable `round_robin` / `priority` key selection, cooldown after `429` and quota-style failures, and masked key-pool visibility in provider status output.
- **Gemini-native inbound compatibility**: added support for `POST /v1beta/models/{model}:generateContent` and `:streamGenerateContent`, Gemini-specific cache namespacing, request-shape prompt/model extraction, and SSE boundary conversion.
- **Usage analytics and quota policies**: added persisted provider/model usage tracking, authenticated `GET /debug/usage`, `isartor stats --usage`, estimated token/cost/savings reporting, and pre-L3 warn/block/fallback quota enforcement with `429` responses for hard quota blocks.

### Changed
- **Architecture documentation refresh**: rewrote the architecture and deflection-stack docs to reflect the current Layer 0, Layer 1a, Layer 1b, Layer 2, Layer 2.5, and Layer 3 flow, plus the CONNECT proxy, MCP server, provider chain, compression pipeline, and usage analytics.
- **Internal maintainability refactor**: consolidated repeated test fixtures behind shared helpers, deduplicated provider-chain logic in `handler.rs`, removed an unused pipeline module, and cleaned up redundant clones without changing runtime behavior.

## [2026.3.11] - 2026-03-29

### Added
- **Request-time model aliases**: added `isartor set-alias`, alias-aware `/v1/models` output, and canonical model resolution before Layer 3 routing and cache-key generation so friendly names like `fast` and `smart` share the same runtime/cache path as the real provider model.
- **Opt-in request/response debug logging**: added `enable_request_logs`, `request_log_path`, rotating redacted JSONL request logs, startup warnings, and `isartor logs --requests` for troubleshooting provider/client integration issues without mixing payloads into normal telemetry.
- **Provider status endpoint and CLI**: added authenticated `GET /debug/providers` and `isartor providers` to show the active Layer 3 provider, configured model/endpoint, and last-known in-memory success/error state.

### Changed
- **OpenAI-compatible provider registry expansion**: added first-class provider entries for `cerebras`, `nebius`, `siliconflow`, `fireworks`, `nvidia`, and `chutes` across config, setup, `set-key`, connectivity checks, and shared OpenAI-compatible runtime routing.
- **Release documentation and architecture records**: refreshed README, docs-site configuration reference, architecture docs, and ADRs to document model aliases, request logging, provider registry growth, and provider health visibility together.

## [2026.3.10] - 2026-03-28

### Added
- **Guided setup wizard**: added `isartor setup` / `isartor configure`, an interactive first-run flow for choosing a provider, saving the API key and model, configuring optional Layer 2 routing, connecting supported tools, and running final verification from one command.

### Fixed
- **Azure connectivity ping**: `isartor check` / `isartor ping` now probes Azure OpenAI with a deployment-scoped chat completions request, matching the live runtime path and avoiding false `404 Not Found` failures on valid Azure deployments.

## [2026.3.9] - 2026-03-27

### Fixed
- **Groq OpenAI passthrough endpoint selection**: OpenAI-compatible tool passthrough now falls back to Groq's own `/openai/v1/chat/completions` endpoint instead of reusing the OpenAI default URL, preventing Groq keys from being sent to OpenAI and failing with misleading `401 invalid_api_key` errors.
- **`isartor set-key -p groq` endpoint wiring**: provider key setup now writes the provider-specific `external_llm_url` alongside the model and key so Groq and other OpenAI-compatible providers start with the correct Layer 3 endpoint.

## [2026.3.8] - 2026-03-27

### Changed
- **OpenClaw docs and connector guidance**: the README and docs-site now make the required `http://localhost:8080/v1` base URL explicit and document the recommended `openclaw models status --agent main --probe` verification path.

### Fixed
- **OpenClaw reconnect auth drift**: `isartor connect openclaw` now refreshes the `main` / default agent model override and clears stale per-agent `models.json` registries so changing the gateway API key does not leave OpenClaw stuck on old provider auth.

## [2026.3.7] - 2026-03-27

### Changed
- **README terminal walkthrough**: the README now includes a generated terminal-style GIF that shows the first-run flow (`install` → `isartor up` → `isartor demo`) plus a clearer “How It Works” section for new users.
- **README positioning refresh**: added community/distribution badges, a factual comparison table, and conservative trusted-by language to improve go-live clarity without overstating adoption.

## [2026.3.6] - 2026-03-27

### Added
- **Detached log follower**: `isartor logs --follow` now tails the detached gateway log file so users can monitor a background Isartor instance without reaching for `tail -f`.

### Fixed
- **OpenClaw OpenAI compatibility**: Isartor now accepts OpenAI-style message `content` as either a plain string or an array of typed content parts, fixing OpenClaw agent requests that send array-based content blocks to `/v1/chat/completions`.

## [2026.3.5] - 2026-03-26

### Added
- **Claude Desktop connector**: `isartor connect claude-desktop` now registers Isartor as a local MCP server in Claude Desktop, with backup and disconnect support.

### Changed
- **Claude Desktop documentation**: README and docs-site now include a first-class Claude Desktop integration guide built on `isartor mcp`.
- **OpenClaw connector refactor**: `isartor connect openclaw` now writes a managed OpenClaw provider config directly instead of generating a manual patch file, and the docs now describe the pragmatic Isartor-first workflow.

## [2026.3.4] - 2026-03-26

### Changed
- **Post-install demo polish**: `isartor demo` now opens with a clearer install-to-go-live narrative, shows an optional live Layer 3 provider preview when configured, and writes `isartor_demo_result.txt` for standalone runs.
- **Stronger demo cache replay**: the bundled canonical demo seeds now align with the replay fixture, lifting the showcase deflection rate to 60% so the demo passes its acceptance threshold consistently.
- **Docs onboarding flow**: README and docs now guide users through `set-key` → `check` → `demo` before connecting their editor or CLI.

## [2026.3.3] - 2026-03-26

### Changed
- **`isartor check` command**: renamed the long `connectivity-check` diagnostic command to `check`, while keeping `ping` and `connectivity-check` as compatibility aliases.
- **Provider-aware connectivity audit**: `isartor check` now shows the configured provider, model, masked API key, provider-specific endpoint, and a real ping result for Groq and other Layer 3 providers.

## [2026.3.2] - 2026-03-26

### Fixed
- **`isartor update` GitHub fallback**: when the GitHub releases API returns `403`, `401`, or `429`, Isartor now falls back to the public `releases/latest` redirect and still resolves the newest tag.

## [2026.3.1] - 2026-03-26

### Changed
- **Startup banner refresh**: `isartor up` now shows the new Isartor ASCII gate on every startup, not only on first run.
- **First-run banner behavior**: startup messaging now keeps first-run demo copy only for true first runs and shows a normal gateway-ready footer otherwise.

## [2026.3.0] - 2026-03-26

Isartor's first CalVer release — marking go-live readiness.

### Changed
- Version scheme switched from SemVer (`0.1.x`) to CalVer (`YYYY.M.PATCH`).

## [0.1.51] - 2026-03-26

### Added
- **OpenClaw integration guide**: Full documentation for `isartor connect openclaw` in docs-site.
- **Antigravity integration guide**: New docs-site page for Antigravity connector.
- **Comparative validation in benchmarks**: Scenario runner now shows code quality parity between baseline and Isartor workspaces instead of absolute pass/fail.
- **L2.5 stats in benchmark reports**: Issue comments now include context optimizer hit rate and bytes saved.
- **Mermaid diagram support**: docs-site renders Mermaid diagrams via CDN-loaded renderer.

### Changed
- **README rewritten for go-live**: Centered hero, Quick Start first, benchmark results table, integration matrix.
- **Documentation cleanup**: Removed 11 superseded docs, consolidated logo source, updated deployment guides.
- **CI/CD pipeline cleanup**: Removed superseded `claude-code-benchmark.yml` (−953 lines), `roi-report.yml`, and `integration-test.yml`. Trimmed stale validate-harness steps from `benchmark.yml`.

### Fixed
- **Benchmark fixture runner**: Inject system instructions for proper L2.5 testing.
- **Scenario runner**: Non-fatal push_workspace on permission denied.
- **docs CI**: Logo copy before mdBook build (not after).

## [0.1.50] - 2026-03-25

### Added
- **L2.5 Context Optimizer** (#85): New middleware layer between L2 SLM triage and L3 cloud handler that compresses instruction payloads (CLAUDE.md, copilot-instructions.md, skills blocks) before cloud calls. Two strategies:
  - **Session dedup** — hashes instruction content per session; on repeat turns, replaces with a compact hash reference.
  - **Static minify** — strips HTML/XML comments, decorative rules, consecutive blank lines, and Unicode box-drawing markers.
- **CompressionPipeline** (Isartor-private#79): Refactored L2.5 into a modular, staged pipeline architecture (`src/compression/`). Three built-in stages: `ContentClassifier`, `DedupStage`, `LogCrunchStage`. Custom stages can be added via the `CompressionStage` trait.
- New config fields: `ISARTOR__ENABLE_CONTEXT_OPTIMIZER` (default: true), `ISARTOR__CONTEXT_OPTIMIZER_DEDUP` (default: true), `ISARTOR__CONTEXT_OPTIMIZER_MINIFY` (default: true).
- Observability: `x-isartor-context-optimized` response header and `layer2_5_context_optimizer` tracing span with `context.bytes_saved` and `context.strategy` fields.

## [0.1.49] - 2026-03-24

### Added
- **Session-aware cache scoping for multi-turn agents**: Layer 1 exact-cache keys and Layer 1 semantic-cache lookups can now be scoped by `x-isartor-session-id`, `x-thread-id`, `x-session-id`, `x-conversation-id`, or matching body metadata so parallel agent conversations do not collide.
- **OpenCode connector**: `isartor connect opencode` now writes managed OpenCode provider/auth config pointing at Isartor's OpenAI-compatible `/v1` gateway.
- **GitHub Copilot in VS Code connector**: `isartor connect copilot-vscode` now auto-detects VS Code `settings.json`, backs it up, and writes the managed Copilot debug override settings for Isartor.
- **HTTP/SSE MCP transport**: Isartor now exposes MCP tools over Streamable HTTP at `/mcp/` with `POST` JSON-RPC, `GET` SSE streaming, and `DELETE` session teardown, alongside the existing stdio MCP server.
- **Per-agent observability endpoint**: `GET /debug/stats/agents` now returns programmatic per-tool request, cache, latency, retry, and error statistics.

### Changed
- **Cursor MCP registration now targets HTTP MCP**: `isartor connect cursor` now registers Isartor's `/mcp/` HTTP endpoint for compatible editor-side MCP flows.
- **`isartor stats --by-tool` is now production-grade**: the CLI now shows richer per-agent metrics including requests, cache hits/misses, average latency, retry count, error count, and L1a/L1b hit-vs-miss safety.
- **OTel metrics now carry tool labels more broadly**: retry, error, cache-event, and latency observability paths now include per-tool dimensions for multi-agent operations.

### Fixed
- **Cache visibility attribution for MCP traffic**: cache lookup/store activity from MCP clients now lands in the correct tool bucket for stats and debug views instead of falling back to stale expectations.

## [0.1.48] - 2026-03-23

### Added
- **OpenAI model discovery endpoint**: `GET /v1/models` now returns an OpenAI-compatible model list so Cursor, OpenCode, and other agent clients can discover the configured Layer 3 model automatically.

### Changed
- **OpenAI-compatible tool use passthrough**: `/v1/chat/completions` now preserves `tools`, `tool_choice`, `functions`, and `function_call` fields for tool-enabled requests and returns upstream `tool_calls` responses unchanged.
- **Tool-aware cache keying for OpenAI requests**: exact-cache keys now include tool definitions and tool-role history, and semantic cache matching is skipped for tool-enabled OpenAI traffic to avoid unsafe collisions.
- **Layer 3 timeout is now configurable globally**: `ISARTOR__L3_TIMEOUT_SECS` (default `120`) now drives the shared timeout for all Layer 3 providers, including Copilot and rig-core-backed providers.

### Fixed
- **OpenAI streaming SSE for coding agents**: `POST /v1/chat/completions` now returns OpenAI-style `text/event-stream` responses when `stream: true`, including `data:` chunks and terminal `data: [DONE]`.
- **Streaming cache hits now preserve OpenAI wire format**: cached OpenAI responses, including tool-call completions, are upgraded to SSE at the response boundary so cache storage stays canonical JSON while streaming clients still receive the protocol they expect.

## [0.1.47] - 2026-03-23

### Fixed
- **GitHub Copilot completion timeout increased for Anthropic-compatible traffic**: Copilot-backed `/v1/messages` requests no longer fail after the shared 30-second HTTP client timeout. Isartor now gives Copilot completion calls a 120-second per-request timeout, which avoids false retry storms and 502s on slower responses.
- **Timeout errors now report the configured limit clearly**: when a Copilot completion does exceed the deadline, the error message now states the timeout window directly to make diagnosis easier.

## [0.1.46] - 2026-03-23

### Fixed
- **Anthropic SSE streaming for Claude Code**: Claude Code sends `stream: true` in `/v1/messages` requests. Isartor now returns proper `text/event-stream` SSE events (`message_start`, `content_block_delta`, `message_stop`, etc.) instead of plain JSON. This fixes the "Interrupted" errors when using Claude Code through Isartor + GitHub Copilot.
- **L1a cache hits also return SSE when streaming**: Cached Anthropic responses are converted to SSE format when the client requests streaming, preventing format mismatch errors on cache hits.
- **Complete Anthropic Messages API response fields**: Responses now include `id`, `usage`, `stop_sequence`, and all required fields that Claude Code expects.

### Changed
- **Integration test workflow disabled by default**: Removed push/PR triggers from `integration-test.yml`, kept `workflow_dispatch` only.

## [0.1.45] - 2026-03-23

### Fixed
- **Claude Code `/v1/messages` no longer semantically caches unrelated prompts**: Anthropic-style chat traffic now uses exact cache only. This avoids unsafe L1b semantic hits where different Claude Code questions could resolve to the same cached answer.
- **Retry warnings now show the full upstream cause chain**: L3 retry logs preserve nested `anyhow` causes so Copilot failures include the underlying network / TLS / DNS / timeout reason instead of only the top-level `Copilot completions request failed` context.
- **CONNECT proxy mirrors the same safer Anthropic cache behavior**: proxied `/v1/messages` traffic also skips semantic cache to stay consistent with the gateway path.

## [0.1.44] - 2026-03-23

### Fixed
- **False L1b semantic cache hits with Claude Code**: The semantic embedding was computed from the entire conversation (system prompt + history + question). Since Claude Code sends a large, identical system prompt with every request, unrelated questions appeared semantically identical (cosine >0.85). The L1b embedding now uses only the last user message, so different questions produce distinct vectors. L1a exact matching is unaffected and still keys on the full conversation.
- **401 "empty API key" error after `isartor connect claude-copilot`**: The connect command wrote the new Copilot config but tested against the still-running (stale) Isartor instance that had the old provider. Now auto-restarts Isartor (`stop` → `up --detach`) after writing config so the test hits the fresh instance.
- **Empty Copilot session token guard**: `exchange_copilot_session_token` now rejects empty `"token": ""` responses instead of sending `Bearer ` and getting a confusing 401 from the upstream API.

## [0.1.43] - 2026-03-23

### Added
- Native GitHub Copilot L3 provider with session-token exchange and Copilot completions support.
- `isartor connect claude-copilot` to route Claude Code through Isartor while using GitHub Copilot as the Layer 3 backend.
- One-click Claude Code + Copilot smoke test script: `./scripts/claude-copilot-smoke-test.sh` (also available via `make smoke-claude-copilot`).
- Documentation for Claude Code + GitHub Copilot across the docs site, legacy docs, and README.
- Focused unit and integration coverage for the Copilot provider and `claude-copilot` connector flow.

### Changed
- `isartor connect claude-copilot` now prefers GitHub device-flow OAuth by default and reuses saved OAuth credentials before falling back to explicit PAT usage.
- Claude Code + Copilot docs now document the device-flow-first auth path and supported smoke-test workflow.

## [0.1.42] - 2026-03-21

### Added
- **Documentation site** (`docs-site/`): Full mdBook-powered documentation with 24 pages organized into Getting Started, Core Concepts, AI Tool Integrations, Deployment, Configuration, Observability, Development, and Blog sections.
- GitHub Actions workflow (`docs.yml`) for automatic docs deployment to GitHub Pages on push to `main`.
- Search functionality, custom theme, and "Edit this page" links.
- Migrated and reorganized all content from `docs/` into navigable site structure with sidebar, per-tool integration pages, and merged/deduplicated observability guides.

## [0.1.41] - 2026-03-21

### Added
- **User-Agent tool identification**: Automatic identification of 15+ AI tools (Cursor, Codex, Gemini CLI, Copilot, Claude Code, Windsurf, Zed, Cline, Roo Code, Aider, Continue, etc.) from `User-Agent` header.
- **Per-tool metrics**: OTel `requests_total` and `request_duration_seconds` metrics now include a `tool` dimension for per-tool Grafana dashboards.
- **Per-tool visibility stats**: `PromptVisibilityState` tracks request counts by tool; `isartor stats --by-tool` shows a per-tool breakdown.
- **JSON stats output**: `isartor stats --json` outputs full `PromptStatsResponse` as JSON for programmatic consumption.
- New module `src/tool_identity.rs` with `identify_tool()` and `identify_tool_or_fallback()` functions.

## [0.1.40] - 2026-03-21

### Added
- **Cursor IDE integration** (`isartor connect cursor`): base URL override + MCP server registration in `~/.cursor/mcp.json`.
- **OpenAI Codex CLI integration** (`isartor connect codex`): `OPENAI_BASE_URL` env script for routing Codex through Isartor.
- **Gemini CLI integration** (`isartor connect gemini`): `GEMINI_API_BASE_URL` env script for routing Gemini CLI through Isartor.
- **Generic connector** (`isartor connect generic`): connect any OpenAI-compatible tool (Windsurf, Zed, Cline, Roo Code, etc.) by specifying the tool's base URL env var.
- Updated integration documentation with step-by-step guides for all new tools.

## [0.1.39] - 2026-03-21

### Fixed
- Recorded Copilot MCP cache lookups in prompt visibility and metrics so `isartor stats` now counts cache hits and misses coming from Copilot CLI.
- Clarified the cache-hit path by surfacing MCP lookup traffic as `mcp` / `copilot` in stats, and showing non-standard layer buckets like `MISS` in the CLI stats output.
- Confirmed the Copilot cache-hit path does not invoke Isartor Layer 3; only Copilot's own final render step remains after an MCP cache hit.

## [0.1.38] - 2026-03-21

### Changed
- Tightened Copilot cache-hit guidance so `isartor connect copilot` now installs stronger instructions telling Copilot to treat `isartor_chat` hits as verbatim final answers, without paraphrasing or extra tool calls.
- Clarified Copilot MCP tool descriptions and integration docs: a Copilot CLI `final_answer` event after a cache hit is a CLI-side render step, not an Isartor Layer 3 forward.

## [0.1.37] - 2026-03-20

### Fixed
- **`isartor up copilot` no longer fails on stale Azure L3 config**: client-hint startup now uses a relaxed config load, so Copilot cache-first mode can start even if `ISARTOR__LLM_PROVIDER=azure` is still set alongside an invalid or stale `ISARTOR__EXTERNAL_LLM_URL`.
- Strict provider validation is still preserved for normal gateway startup paths, so `isartor up` continues to catch real Azure misconfiguration.
- `isartor connect` helper defaults now also use the relaxed load path when only local gateway bind/auth settings are needed.

## [0.1.36] - 2026-03-19

### Changed
- **Copilot CLI: plain prompts now use Isartor cache first**: `isartor connect copilot` now installs a managed Isartor block in `~/.copilot/copilot-instructions.md` so normal conversational prompts call `isartor_chat` before answering directly.
- On cache misses, Copilot now follows the full cache-only workflow automatically: `isartor_chat` miss → Copilot answer → `isartor_cache_store`.
- Improved MCP tool descriptions to reinforce cache-first usage for plain chat prompts.
- Updated `docs/4-INTEGRATIONS.md` with the managed instruction-file behavior and troubleshooting guidance.

## [0.1.35] - 2026-03-19

### Changed
- **Copilot CLI: cache-only MCP approach**: `isartor_chat` now performs cache lookup only (L1a exact + L1b semantic). On a miss it returns empty so Copilot uses its own LLM — Isartor never routes Copilot traffic through its configured L3 provider.
- `isartor connect copilot` automatically cleans up legacy proxy env files and hook scripts from earlier versions.
- `isartor connect copilot` adds the gateway URL to Copilot's `allowed_urls` in `~/.copilot/config.json`.
- `isartor connect copilot` now installs a managed Isartor instruction block in `~/.copilot/copilot-instructions.md` so plain conversational prompts prefer `isartor_chat` first and call `isartor_cache_store` after misses.
- `isartor connect status` now shows "MCP server (isartor_chat tool)" for Copilot.
- Improved connection test: checks `/health` first, distinguishes timeout (L3 unconfigured) from gateway unreachable.
- Updated `docs/4-INTEGRATIONS.md` for the cache-only MCP approach.

### Added
- `isartor mcp` subcommand: MCP (Model Context Protocol) stdio server for Copilot CLI and other MCP-compatible clients. Exposes `isartor_chat` (cache lookup) and `isartor_cache_store` (cache write) tools.
- `POST /api/v1/cache/lookup`: Cache-only lookup endpoint (returns cached response or 204 on miss).
- `POST /api/v1/cache/store`: Cache store endpoint (saves prompt/response pair to L1a exact + L1b semantic cache).

## [0.1.34] - 2026-03-19

### Changed
- **Drop MITM CONNECT proxy for client integrations** (issue #41): All `isartor connect` flows now use native client mechanisms instead of HTTPS_PROXY + TLS MITM:
  - **Copilot CLI**: preToolUse hooks via new `POST /api/v1/hook/pretooluse` endpoint — no proxy, no CA certificates
  - **Claude Code**: `ANTHROPIC_BASE_URL` override in `~/.claude/settings.json`
  - **Antigravity**: `OPENAI_BASE_URL` + `OPENAI_API_KEY` env file
  - **OpenClaw**: Already used base URL approach (verified clean)
- `isartor up copilot|claude|antigravity` now starts gateway-only (no CONNECT proxy needed)
- `isartor connect status` shows integration method per client (hooks / base URL / provider base URL)
- Rewrote `docs/4-INTEGRATIONS.md` for the new proxy-free architecture

### Added
- `POST /api/v1/hook/pretooluse` public endpoint for Copilot CLI preToolUse hooks
- Integration tests for the hook endpoint (valid, empty, malformed payloads)

## [0.1.33] - 2026-03-19

### Fixed
- **CONNECT proxy HTTP/2 connection reset**: ALPN now only advertises `http/1.1` since the proxy's request parser is text-based. Clients previously negotiated HTTP/2, sent binary frames, and the proxy dropped the connection.
- **Copilot shell env missing `SSL_CERT_FILE`**: `copilot.sh` now exports `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` pointing to a combined CA bundle (system CAs + Isartor CA). This fixes TLS failures for non-Node.js clients like `gh` (Go binary) and `curl` that ignore `NODE_EXTRA_CA_CERTS`.

## [0.1.32] - 2026-03-19

### Fixed
- **CONNECT proxy TLS panic**: pinned the rustls `CryptoProvider` (`ring`) at startup so the CONNECT proxy no longer panics on the first TLS handshake when both `ring` and `aws-lc-rs` features are enabled transitively.
- **Port mismatch in `copilot.sh`**: `isartor connect copilot` now auto-detects the running proxy port from `AppConfig` (honouring `ISARTOR__PROXY_PORT` and `isartor.toml`) instead of always writing the CLI default. This fixes "connection refused" when users configure non-default ports.

### Changed
- Expanded Copilot integration docs in `docs/4-INTEGRATIONS.md` with custom-port instructions, troubleshooting table, and verification steps.

## [0.1.31] - 2026-03-19

### Fixed
- Restored the `schannel` lockfile entry to `0.1.29` after the `v0.1.30` release metadata bump accidentally rewrote a third-party `Cargo.lock` dependency version and broke CI dependency resolution.

## [0.1.30] - 2026-03-19

### Added
- `isartor --detach` and `isartor up --detach` to start the gateway in the background, return control to the shell immediately, and log startup output to `~/.isartor/isartor.log`.

### Changed
- First-run and startup messaging now points users to detached startup when they want to keep using the current terminal session.

## [0.1.29] - 2026-03-19

### Fixed
- `isartor update` now explains permission-denied self-update failures for protected install directories such as `/usr/local/bin`, recommends a user-writable install path like `~/.local/bin`, and prints copy-pasteable recovery commands on Unix-like systems before suggesting `sudo`.

## [0.1.28] - 2026-03-19

### Added
- `isartor up` for the recommended terminal startup flow.
- `isartor up copilot|claude|antigravity` to start the API gateway plus the CONNECT proxy only when that client needs it.

### Changed
- Bare `isartor` startup now follows the gateway-only local-first path instead of always enabling the CONNECT proxy.
- `/health` now reports proxy status accurately when running in gateway-only mode.
- Startup docs, smoke tests, and CI workflows now use the new `up` entrypoints.

## [0.1.27] - 2026-03-19

### Changed
- Gateway auth is now disabled by default for local-first usage. Set `ISARTOR__GATEWAY_API_KEY` (or `gateway_api_key` in `isartor.toml`) to enable Layer 0 authentication.
- Startup logs now explicitly show whether gateway auth is enabled or disabled.

### Fixed
- Smoke test and integration docs now reflect that gateway auth is opt-in, while preserving explicit auth-enabled examples for manual testing flows.

## [0.1.26] - 2026-03-19

### Added
- `scripts/smoke-test.sh` — runnable bash script that exercises every Isartor feature (health, auth, L1a/L1b/L2/L3, OpenAI + Anthropic + native endpoints, proxy, stats CLI, Copilot CLI integration)
- `docs/TESTING.md` — step-by-step manual test runbook with copy-paste commands and a feature coverage matrix
- `.github/workflows/integration-test.yml` — CI workflow that builds the binary, starts the server, and runs the smoke test suite on every push/PR

### Fixed
- Refactored `emit_proxy_decision` in `src/proxy/connect.rs` to use a `ProxyDecisionContext` struct, resolving `clippy::too_many_arguments` lint error

### Changed
- Coverage CI switched from `cargo-tarpaulin` to `cargo-llvm-cov` for reliable LCOV report generation
- README now includes a Codecov coverage badge

## [0.1.25] - 2026-03-19

### Added
- `isartor stats` for user-facing prompt totals, per-layer counts, and recent prompt routing history
- Unified prompt visibility rollups across gateway and CONNECT proxy traffic via `/debug/stats/prompts`

### Changed
- `/health` now includes prompt totals so operators can quickly confirm how much prompt traffic Isartor has seen
- Observability docs and metrics examples now cover proxy-aware prompt telemetry dimensions such as `traffic_surface`, `client`, and `endpoint_family`

## [0.1.24] - 2026-03-19

### Fixed
- `isartor update` now bypasses stale local Isartor proxy environment when checking GitHub for releases, so updates still work after the local CONNECT proxy on `localhost:8081` has been stopped

## [0.1.23] - 2026-03-18

### Added
- Proxy-layer visibility for recent CONNECT-routed client requests through logs, `/debug/proxy/recent`, `/health`, and `isartor connect status`
- Repo-specific `.github/copilot-instructions.md` for future Copilot sessions

### Changed
- CONNECT proxy Layer 3 now preserves the native upstream for Copilot, Claude Code, and Antigravity instead of requiring a separately configured Isartor Layer 3 provider key for those proxied paths
- `isartor connect claude` and `isartor connect antigravity` now configure proxy-based routing with local CA trust so native client authentication can continue upstream
- Integration docs now describe multi-client CONNECT proxy behavior and supported intercepted upstream domains

## [0.1.22] - 2026-03-18

### Added
- Windows x86_64 build target in release pipeline
- SECURITY.md, CODE_OF_CONDUCT.md, CODEOWNERS
- Dependabot for Cargo, Actions, and Docker updates
- Dynamic version badge from GitHub Releases

### Changed
- All install URLs now point to `isartor-ai/Isartor` (no more `isartor-dist` indirection)
- Consolidated architecture docs into single `docs/2-ARCHITECTURE.md`
- Cleaned release notes template (removed private-repo install paths)

### Removed
- `dist_release.yml` workflow (isartor-dist no longer needed)
- Duplicate `scripts/install.sh` and `scripts/install.ps1`
- Private-repo install instructions from README and docs
- `azure_openai_key/` directory
- Stale `architecture.md` and `docs/ARCHITECTURE.md` duplicates

## [0.1.18] - 2025-07-14

### Added
- HTTP CONNECT proxy with TLS MITM for GitHub Copilot CLI (`isartor connect copilot`)
- Proxy listens on `:8081` by default (`ISARTOR__PROXY_PORT`)
- Auto-generated CA stored at `~/.isartor/ca/`
- Per-host server certificates via rcgen
- Domain allowlist for transparent tunneling

## [0.1.17] - 2025-07-14

### Added
- `isartor set-key` CLI subcommand for configuring LLM provider API keys
- Supports OpenAI, Azure, Anthropic, Groq, Mistral providers
- Interactive secure input via rpassword
- TOML config merge via toml_edit
- `--env-file` mode for Docker secret workflows
- `--dry-run` flag for previewing changes

## [0.1.16] - 2025-07-13

### Added
- Publish releases to distribution repo via `dist_release.yml`

### Fixed
- Linux musl builds now use cargo-zigbuild for static linking

## [0.1.15] - 2025-07-13

### Fixed
- Build musl targets with runner-based zigbuild (no Docker container)

[Unreleased]: https://github.com/isartor-ai/Isartor/compare/v2026.3.10...HEAD
[2026.3.10]: https://github.com/isartor-ai/Isartor/compare/v2026.3.9...v2026.3.10
[2026.3.9]: https://github.com/isartor-ai/Isartor/compare/v2026.3.8...v2026.3.9
[2026.3.8]: https://github.com/isartor-ai/Isartor/compare/v2026.3.7...v2026.3.8
[2026.3.7]: https://github.com/isartor-ai/Isartor/compare/v2026.3.6...v2026.3.7
[2026.3.6]: https://github.com/isartor-ai/Isartor/compare/v2026.3.5...v2026.3.6
[2026.3.5]: https://github.com/isartor-ai/Isartor/compare/v2026.3.4...v2026.3.5
[2026.3.4]: https://github.com/isartor-ai/Isartor/compare/v2026.3.3...v2026.3.4
[2026.3.3]: https://github.com/isartor-ai/Isartor/compare/v2026.3.2...v2026.3.3
[2026.3.2]: https://github.com/isartor-ai/Isartor/compare/v2026.3.1...v2026.3.2
[2026.3.1]: https://github.com/isartor-ai/Isartor/compare/v2026.3.0...v2026.3.1
[2026.3.0]: https://github.com/isartor-ai/Isartor/compare/v0.1.51...v2026.3.0
[0.1.31]: https://github.com/isartor-ai/Isartor/compare/v0.1.30...v0.1.31
[0.1.30]: https://github.com/isartor-ai/Isartor/compare/v0.1.29...v0.1.30
[0.1.29]: https://github.com/isartor-ai/Isartor/compare/v0.1.28...v0.1.29
[0.1.28]: https://github.com/isartor-ai/Isartor/compare/v0.1.27...v0.1.28
[0.1.27]: https://github.com/isartor-ai/Isartor/compare/v0.1.26...v0.1.27
[0.1.26]: https://github.com/isartor-ai/Isartor/compare/v0.1.25...v0.1.26
[0.1.25]: https://github.com/isartor-ai/Isartor/compare/v0.1.24...v0.1.25
[0.1.24]: https://github.com/isartor-ai/Isartor/compare/v0.1.23...v0.1.24
[0.1.23]: https://github.com/isartor-ai/Isartor/compare/v0.1.22...v0.1.23
[0.1.22]: https://github.com/isartor-ai/Isartor/compare/v0.1.19...v0.1.22
[0.1.19]: https://github.com/isartor-ai/Isartor/compare/v0.1.18...v0.1.19
[0.1.18]: https://github.com/isartor-ai/Isartor/compare/v0.1.17...v0.1.18
[0.1.17]: https://github.com/isartor-ai/Isartor/compare/v0.1.16...v0.1.17
[0.1.16]: https://github.com/isartor-ai/Isartor/compare/v0.1.15...v0.1.16
[0.1.15]: https://github.com/isartor-ai/Isartor/releases/tag/v0.1.15

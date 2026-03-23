# Changelog

All notable changes to Isartor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/isartor-ai/Isartor/compare/v0.1.31...HEAD
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

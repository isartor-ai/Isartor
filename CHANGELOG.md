# Changelog

All notable changes to Isartor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

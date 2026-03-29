# Copilot Instructions for Isartor

## Build, test, and lint commands

- Use `cargo build` for a local build and `cargo build --release` or `make build` for a release build.
- Run the full test suite with `cargo test --all-features` or `make test`.
- Run formatting and lint checks the same way CI does:
  - `cargo fmt --all -- --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
- Run a single test by targeting the integration-test binary that owns it, for example:
  - `cargo test --test scenario_suite deflection_rate_at_least_60_percent -- --nocapture`
  - `cargo test --test integration_suite body_survives_all_middleware -- --nocapture`
  - `cargo test --test unit_suite exact_cache_miss_then_hit -- --nocapture`
  - `cargo test --test integration_test redis_exact_cache_round_trip -- --ignored --nocapture`
- Run Criterion benches with the explicit bench target, for example:
  - `cargo bench --bench cache_latency`
  - `cargo bench --bench e2e_pipeline`
- Run the benchmark harness with:
  - `make benchmark`
  - `make benchmark-dry-run`
  - `python3 benchmarks/run.py --url http://localhost:8080 --input benchmarks/fixtures/faq_loop.jsonl --requests 500`
- Development setup in `CONTRIBUTING.md` expects Rust 1.75+ and notes Docker for integration-test workflows.
- Update tests in `tests/`, targeting the owning integration-test binary (`unit_suite`, `integration_suite`, `scenario_suite`, or `integration_test`) rather than assuming per-file test targets.
- Treat `README.md` as the public feature list and top-level product entry point.
- For any implementation ticket that changes behavior, capabilities, supported providers/tools, CLI/API surfaces, configuration, or UX, update the affected documentation in the same change:
  - `README.md` feature list and user-facing capability callouts
  - supplementary docs in `docs/`
  - published docs in `docs-site/src/`
- If the implementation changes system structure, request flow, supported surfaces, deployment model, routing strategy, cache semantics, or other lasting technical trade-offs, also update:
  - `docs/architecture-decisions.md`
  - `docs-site/src/concepts/architecture-decisions.md`
  - `docs-site/src/concepts/architecture.md`
- Do not defer these documentation or architecture updates to a follow-up ticket when the implementation itself is shipping now.
## High-level architecture

- `src/main.rs` is the real boot sequence. It loads `AppConfig`, initializes telemetry, eagerly loads the in-process sentence embedder, builds a shared `AppState`, starts the Axum API gateway, starts a separate CONNECT proxy, and optionally runs the first-run demo.
- The main HTTP request path is a deflection stack implemented as Axum middleware plus a final handler. For authenticated routes the intended execution order is:
  - body buffer
  - request-level monitoring
  - auth
  - Layer 1 cache
  - Layer 2 SLM triage
  - Layer 2.5 context optimizer
  - Layer 3 handler
- That order is easy to break because Axum middleware wraps inside-out: the last `.layer(...)` added runs first. `src/main.rs` documents this explicitly and the stack should stay in that order.
- Public health routes (`/health`, `/healthz`) intentionally bypass the deflection stack. The authenticated routes are `/api/chat`, `/api/v1/chat`, `/v1/chat/completions`, and `/v1/messages`.
- `src/state.rs` is the shared runtime wiring. `AppState` owns the exact cache, semantic vector cache, text embedder, SLM client, and the configured Layer 3 provider wrapped as `AppLlmAgent`.
- The codebase uses a hexagonal / ports-and-adapters design for swappable backends. The abstractions live in `src/core/ports.rs`, implementations live in `src/adapters/`, and `src/factory.rs` shows how memory-vs-Redis and embedded-vs-vLLM choices are intended to be selected from config.
- There are two request-processing surfaces:
  - The Axum gateway for native, OpenAI-compatible, and Anthropic-compatible HTTP APIs.
  - The CONNECT MITM proxy in `src/proxy/connect.rs`, which intercepts allowlisted GitHub Copilot domains and reuses the cache/LLM layers for `/v1/chat/completions` traffic.

## Key conventions

- Configuration is driven by `ISARTOR__...` environment variables with double underscores. Nested fields also use double underscores, for example `ISARTOR__LAYER2__SIDECAR_URL`. Do not switch these to single-underscore variants.
- Preserve response shape by endpoint. The code intentionally keeps cache namespaces separate for native, OpenAI, Anthropic, and proxy flows so one endpoint does not return another endpoint's schema.
- Cache keys are namespaced before hashing (`native|prompt`, `openai|prompt`, `anthropic|prompt`, etc.). Keep that behavior when adding routes or cache lookups.
- Layer 1 cache writes normalize `ChatResponse.layer` to `1` before storing. A response originally produced by Layer 2 or Layer 3 should still report Layer 1 on a cache hit.
- Layer 3 stale fallback first checks the namespaced exact-cache key, then a legacy un-namespaced key for backward compatibility.
- `BufferedBody` is a hard requirement for the stack. Middleware and handlers read the buffered body from request extensions instead of consuming the stream directly; there is a dedicated integration test proving the body survives all layers.
- `enable_slm_router = false` means Layer 2 is a no-op and the request falls through to Layer 3. `offline_mode = true` blocks Layer 3 cloud routing explicitly instead of silently pretending success.
- Tests are grouped into integration-test binaries (`unit_suite`, `integration_suite`, `scenario_suite`) that just re-export submodules. When running or updating a test, target the owning `--test` binary rather than assuming the function lives in a standalone file target.
- The benchmark harness is part of the repo contract, not just ad hoc tooling. `make benchmark` expects a live Isartor instance plus a matching `X-API-Key`, and `make benchmark-dry-run` is the no-server smoke test used in CI.

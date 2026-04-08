# Isartor — Claude Code Guide

This file helps Claude Code understand the Isartor codebase quickly and make safe changes.

## What Isartor is

Isartor is a pure-Rust LLM gateway / prompt firewall. It sits in front of coding tools and SDKs, tries to resolve requests locally, and only sends the hard ones to a cloud model.

Main request flow:

`L1a exact cache -> L1b semantic cache -> L2 SLM triage -> L3 cloud provider`

Supported client surfaces:

- Native API: `POST /api/chat`
- OpenAI-compatible API: `POST /v1/chat/completions`
- Anthropic-compatible API: `POST /v1/messages`
- Gemini-compatible API: `POST /v1beta/models/*:generateContent`
- Cursor / Kiro compatibility on `/v1/chat/completions` via header-based format detection
- CONNECT proxy for Copilot-style traffic: `src/proxy/connect.rs`
- MCP server for tool integrations: `isartor mcp`

## What matters most architecturally

### 1. `src/main.rs` is the real boot sequence

`src/main.rs` loads config, initializes telemetry, builds shared `AppState`, wires the Axum router, and optionally starts the CONNECT proxy.

Authenticated routes are wired in `src/main.rs` and intentionally go through the full middleware stack.

### 2. Middleware ordering is critical

The intended request execution order is:

`body buffer -> monitoring -> auth -> cache -> slm triage -> context optimizer -> handler`

Axum middleware wraps inside-out, so the last `.layer(...)` added runs first. Do not reorder casually.

### 3. `BufferedBody` is required

`src/middleware/body_buffer.rs` reads the request body once and stores a clone in request extensions. Downstream layers are expected to read from `BufferedBody`, not consume the stream directly.

This is especially important for:

- cache key extraction
- Anthropic/OpenAI compatibility handlers
- monitoring
- retries and middleware composition

### 4. `AppState` is the runtime wiring hub

`src/state.rs` owns:

- the shared `reqwest::Client`
- the L1 exact cache
- the semantic vector cache
- the in-process text embedder
- the configured L3 provider via `AppLlmAgent`
- the SLM client

Provider selection is config-driven in `AppState::new()`.

## Key code locations

### API / request handling

- `src/main.rs` — router construction and server boot
- `src/handler.rs` — native, OpenAI-compatible, and Anthropic-compatible handlers
- `src/anthropic_sse.rs` — Anthropic SSE helpers for Claude Code / `/v1/messages`

### Middleware layers

- `src/middleware/body_buffer.rs` — preserves request body
- `src/middleware/monitoring.rs` — root request tracing / metrics
- `src/middleware/auth.rs` — gateway API key auth
- `src/middleware/cache.rs` — L1a exact + L1b semantic cache
- `src/middleware/slm_triage.rs` — L2 classifier / local-answer short circuit
- `src/middleware/context_optimizer.rs` — L2.5 instruction dedup + minification
- `src/compression/` — Modular `CompressionPipeline` with pluggable stages

### Runtime state / providers

- `src/state.rs` — `AppState`, `AppLlmAgent`, provider construction
- `src/auth/` — provider auth flows, encrypted token store, `isartor auth`
- `src/sync/` — encrypted config sync, conflict detection, blob-server routes, `isartor sync`
- `src/providers/` — L3 providers
- `src/providers/copilot.rs` — GitHub Copilot-backed L3 provider
- `src/core/prompt.rs` — stable prompt extraction for cache keys
- `src/core/context_compress.rs` — L2.5 instruction detection, dedup, minification
- `src/errors.rs` — gateway error formatting and error-chain handling

### CONNECT proxy

- `src/proxy/connect.rs` — HTTPS interception flow for Copilot-compatible traffic

### Tests

Tests are grouped by integration-test binary:

- `tests/unit_suite.rs`
- `tests/integration_suite.rs`
- `tests/scenario_suite.rs`
- `tests/integration_test.rs`

When running one test, target the owning test binary, not an imaginary per-file target.

Examples:

```bash
cargo test --test unit_suite exact_cache_miss_then_hit -- --nocapture
cargo test --test integration_suite body_survives_all_middleware -- --nocapture
cargo test --test scenario_suite deflection_rate_at_least_60_percent -- --nocapture
```

## Current important behavior and guardrails

### Anthropic / Claude Code behavior

Claude Code uses `POST /v1/messages`.

Important current rules:

- `/v1/messages` supports Anthropic JSON responses and SSE streaming
- `src/anthropic_sse.rs` is used to convert responses into proper Anthropic SSE when `stream: true`
- L1a exact cache is enabled for `/v1/messages`
- L1b semantic cache is intentionally disabled for `/v1/messages`

Why L1b is disabled there:

- Claude Code sends large, repetitive system/context payloads
- semantic similarity caused false cache hits across different user questions
- exact cache remains safe and useful

If you touch Claude Code or Anthropic compatibility, preserve this behavior unless you are intentionally redesigning it and adding regression tests.

### Cache-key behavior

`src/core/prompt.rs` contains two important functions:

- `extract_prompt()` — full stable prompt extraction used for exact cache keys
- `extract_semantic_key()` — last-user-message extraction used for semantic matching

`src/core/cache_scope.rs` adds optional session/thread-aware cache scoping:

- cache keys stay namespaced by API surface (`native`, `openai`, `anthropic`)
- if a request supplies `x-isartor-session-id`, `x-thread-id`, `x-session-id`, `x-conversation-id`, or body `session_id` / `thread_id` / `conversation_id` metadata, L1a/L1b are additionally scoped to that session
- the session identifier is hashed before it is mixed into cache keys or stored in the semantic cache index
- if no usable session identifier is present, caching keeps the old global behavior

For Anthropic traffic, exact caching still uses the full conversation shape. Semantic matching is not used on `/v1/messages`, and this change does not re-enable L1b there.

### Response-shape separation

Isartor intentionally namespaces cache keys by API surface to avoid returning the wrong schema:

- native
- openai
- anthropic

Do not make cache entries shared across these formats.

### Stored credentials and sync boundaries

- `isartor auth <provider>` stores encrypted credentials under `~/.isartor/tokens/`
- if no explicit provider API key is configured, Layer 3 does a best-effort lookup in that token store
- `isartor sync ...` only syncs the portable subset of `isartor.toml`
- OAuth tokens, cache contents, usage history, and other local runtime state do **not** sync

### L3 stale / compatibility safety

If you change handlers or provider wiring, preserve endpoint-specific response shapes:

- `/api/chat` returns native `ChatResponse`
- `/v1/chat/completions` returns OpenAI-style responses
- `/v1/messages` returns Anthropic-style responses

### Config naming convention

Environment variables use double underscores:

`ISARTOR__...`

Examples:

- `ISARTOR__LLM_PROVIDER`
- `ISARTOR__EXTERNAL_LLM_API_KEY`
- `ISARTOR__LAYER2__SIDECAR_URL`

Do not convert nested config to single-underscore names.

## Build, test, lint

Use the same commands CI expects:

```bash
cargo build
cargo test --all-features
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

If you changed Rust code, run at least:

```bash
cargo fmt --all
cargo build
cargo clippy --all-targets --all-features -- -D warnings
```

## Useful local commands

### Start the gateway

```bash
isartor up
isartor up --detach
```

### Health check

```bash
curl http://localhost:8080/health
```

### Claude Code + Copilot path

```bash
isartor connect claude-copilot
isartor up --detach
claude
```

Smoke test:

```bash
./scripts/claude-copilot-smoke-test.sh
```

### Stats / observability

```bash
isartor stats
isartor stats --by-tool
curl http://localhost:8080/debug/stats/prompts
```

## Patterns to follow when changing code

### Prefer surgical fixes

This repo has a lot of protocol-compatibility logic. Small response-shape or cache-key changes can break clients in subtle ways.

### Add tests for protocol regressions

If you change any of these, add or update tests:

- `/v1/messages`
- SSE behavior
- cache hit behavior
- prompt extraction
- provider error handling
- middleware ordering assumptions

### Preserve explicit errors

Do not add broad silent fallbacks. If a provider call fails, return a clear gateway error and keep the error chain visible in logs.

## Common gotchas

- Middleware order in Axum is easy to break
- `BufferedBody` must be present for downstream logic
- Claude Code requires proper Anthropic response shapes and SSE when streaming
- `/v1/messages` should not use semantic cache
- The shared HTTP client in `AppState` has a general timeout, but provider-specific calls may override it
- The CONNECT proxy and the gateway should stay behaviorally aligned where practical

## If you are adding a feature

Ask:

1. Which API surface does this affect: native, OpenAI, Anthropic, proxy, or MCP?
2. Does it change cache keys or cache safety?
3. Does it preserve response format by endpoint?
4. Does it require tests in `unit_suite`, `integration_suite`, or `scenario_suite`?
5. Does it require updates to the public feature list in `README.md`?
6. Which supplementary docs in `docs/` and published docs in `docs-site/src/` need to change in the same patch?
7. Does it change system structure, request flow, supported surfaces, deployment shape, routing strategy, or another lasting design decision that should update:
   - `docs/architecture-decisions.md`
   - `docs-site/src/concepts/architecture-decisions.md`
   - `docs-site/src/concepts/architecture.md`

For implementation tickets, treat these documentation and architecture updates as part of done-ness, not follow-up work.

## If you are debugging a bug

Start here:

- `src/main.rs`
- `src/handler.rs`
- `src/middleware/cache.rs`
- `src/middleware/slm_triage.rs`
- `src/core/prompt.rs`
- `src/providers/copilot.rs` or the relevant provider
- `~/.isartor/isartor.log` at runtime

That is the shortest path to understanding most user-facing bugs in this project.

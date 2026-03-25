# Contributing

Thanks for your interest in contributing to Isartor! Isartor is maintained
by one developer as a side project. Here's how to make your contribution
land quickly.

---

## Before You Open a PR

1. **Check existing issues** — your idea may already be tracked.
2. **Open an issue first** for any non-trivial change.
3. **One PR per issue** — keep scope tight.

Looking for something to work on? Check out the
[good first issues](https://github.com/isartor-ai/Isartor/labels/good%20first%20issue)
label on GitHub.

---

## Development Setup

### Prerequisites

- **Rust 1.75+** — install via [rustup](https://rustup.rs/)
- **Docker** — required for integration tests and the observability stack
- **curl + jq** — for manual testing

### Clone and Build

```bash
git clone https://github.com/isartor-ai/Isartor.git
cd Isartor
cargo build
```

### Run the Test Suite

```bash
# Full test suite
cargo test --all-features

# Or use Make
make test

# Run a specific test binary
cargo test --test unit_suite
cargo test --test integration_suite
cargo test --test scenario_suite

# Run a single test with output
cargo test --test scenario_suite deflection_rate_at_least_60_percent -- --nocapture
```

### Lint & Format

```bash
# Format check (same as CI)
cargo fmt --all -- --check

# Apply formatting
cargo fmt --all

# Clippy lint check (same as CI)
cargo clippy --all-targets --all-features -- -D warnings
```

### Release Build

```bash
cargo build --release
# or
make build
```

### Benchmarks

```bash
# Criterion micro-benchmarks
cargo bench --bench cache_latency
cargo bench --bench e2e_pipeline

# Full benchmark harness (requires running Isartor instance)
make benchmark

# Dry-run smoke test (no server needed)
make benchmark-dry-run
```

---

## PR Checklist

- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` has no new warnings
- [ ] `cargo fmt --all -- --check` passes
- [ ] PR description explains **WHY**, not just WHAT
- [ ] Documentation updated if behaviour changes

---

## What Gets Merged Quickly

- **Bug fixes** with a test that reproduces the bug
- **Documentation improvements**
- **Performance improvements** with benchmark evidence

## What Takes Longer

- **New features** — needs design discussion in an issue first
- **Changes to the deflection layer logic** — core path changes require careful review

---

## Code Conventions

- Tests are grouped into integration-test binaries (`unit_suite`, `integration_suite`, `scenario_suite`) that re-export submodules. When adding a test, place it in the appropriate binary rather than creating a standalone file.
- Configuration uses `ISARTOR__...` environment variables with **double underscores** as separators.
- The Axum middleware stack wraps inside-out. See `src/main.rs` for the documented layer order.
- Use `spawn_blocking` for CPU-intensive work (embeddings, model inference) to avoid starving the Tokio runtime.
- The `src/compression/` module uses a **Fusion Pipeline** pattern: stateless `CompressionStage` trait objects executed in order. To add a new compression stage, implement the `CompressionStage` trait and wire it in `src/compression/optimize.rs::build_pipeline()`.

---

## Response Time

Issues and PRs are reviewed within 24–48 hours on weekdays. Weekend
responses are not guaranteed.

---

*See also: [Testing](testing.md) · [Architecture](../concepts/architecture.md) · [Troubleshooting](troubleshooting.md)*

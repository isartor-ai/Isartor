# Contributing to Isartor

Thanks for your interest in contributing. Isartor is maintained
by one developer as a side project. Here's how to make your
contribution land quickly.

## Before You Open a PR

1. **Check existing issues** — your idea may already be tracked
2. **Open an issue first** for any non-trivial change
3. **One PR per issue** — keep scope tight

## Development Setup

```bash
git clone https://github.com/isartor-ai/Isartor.git
cd Isartor
cargo build
cargo test --all-features
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Requirements: Rust 1.75+, Docker (for integration tests)

## PR Checklist

- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo fmt --all` applied
- [ ] PR description explains WHY, not just WHAT
- [ ] README / docs / published docs updated if behavior, config, CLI, or architecture changed

## Documentation Expectations

If your change ships new behavior, update the matching docs in the same PR:

- `README.md` for user-facing features and CLI changes
- `docs/` for supplementary reference and architecture notes
- `docs-site/src/` for the published documentation site

If the change affects long-lived system structure or trade-offs, update
`docs/architecture-decisions.md` too.

## What Gets Merged Quickly

- Bug fixes with a test that reproduces the bug
- Documentation improvements
- Performance improvements with benchmark evidence

## What Takes Longer

- New features (needs design discussion in an issue first)
- Changes to the deflection layer logic

## Response Time

I respond to issues and PRs within 24–48 hours on weekdays.
Weekend responses are not guaranteed.

# Contributing to Isartor

First off, thank you for considering contributing to Isartor! It's people like you that make Isartor such a great tool for the AI community.

We welcome all kinds of contributions, whether it's:
- ✨ Implementing new features or Deflection Stack layers.
- 🐛 Reporting or fixing bugs.
- 📝 Improving documentation (including this guide!).
- 🧪 Adding unit or integration tests.
- 💡 Proposing new architectural ideas.

## 🛠️ Prerequisites

To build and test Isartor locally, you will need the following tools:

1.  **Rust Toolchain**: Install via [rustup.rs](https://rustup.rs/). We track the latest stable release.
2.  **CMake**: Required for building some of the ML backends (ort-sys, candle-core).
3.  **Docker & Docker Compose**: Essential for running integration tests, sidecars (llama.cpp), and the observability stack (Prometheus/Grafana).
4.  **Git**: For version control.

## 🚀 Development Workflow

The standard way to contribute is:

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/Isartor.git
    cd Isartor
    ```
3.  **Create a feature branch**:
    ```bash
    git checkout -b feat/your-awesome-feature
    ```
4.  **Make your changes**. Ensure your code adheres to our [Coding Standards](#coding-standards).
5.  **Run the tests**:
    ```bash
    cargo test
    ```
6.  **Run the linter and formatter**:
    We use strict CI checks. Save yourself some time by running these locally:
    ```bash
    cargo fmt --all
    cargo clippy --all-targets --all-features -- -D warnings
    ```
7.  **Commit your changes**. We prefer descriptive, conventional commit messages (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
8.  **Push to your fork** and **submit a Pull Request** against the `main` branch.

## 📏 Coding Standards

- **Idiomatic Rust**: We strive for clean, "idiomatic" Rust. When in doubt, follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- **Formatting**: Always run `cargo fmt` before committing. We use the default `rustfmt` settings.
- **Linting**: We take Clippy warnings seriously. All PRs must pass `cargo clippy` with no warnings.
- **Documentation**: Public modules, structs, and functions should be documented with doc comments (`///`).

## 💬 Community & Communication

Have a big idea or a complex question? We'd love to discuss it before you spend hours on a PR!

- **Discord**: Join our [Discord Server](https://discord.gg/placeholder) for real-time discussion.
- **Matrix**: You can also find us on Matrix at [#isartor:matrix.org](https://matrix.to/#/#isartor:matrix.org).
- **GitHub Issues**: For bug reports and formal feature requests, please use the [Issue Tracker](https://github.com/isartor-ai/Isartor/issues).

---

By contributing to this project, you agree that your contributions will be licensed under the **Apache License, Version 2.0**.

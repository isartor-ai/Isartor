# ── Stage 1: Build ─────────────────────────────────────────────────────
FROM rust:latest AS builder

# cmake and g++ are needed by aws-lc-sys (TLS crypto) and tokenizers.
RUN apt-get update && apt-get install -y cmake g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Optional build-time flags forwarded from Compose (e.g. --no-default-features).
ARG CARGO_OPTS=""

# Copy everything needed for the build.
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
COPY demo/ demo/

# Build the release binary (glibc, default target).
# shellcheck disable=SC2086 - word-splitting of CARGO_OPTS is intentional.
RUN cargo build --release ${CARGO_OPTS} \
    && cp target/release/isartor /build/isartor-bin

# ── Stage 2: Runtime ──────────────────────────────────────────────────
# cc-debian12 includes glibc + libgcc — needed for dynamically linked binaries.
FROM gcr.io/distroless/cc-debian12

COPY --from=builder /build/isartor-bin /isartor

ENV ISARTOR_HOST_PORT=0.0.0.0:8080
EXPOSE 8080

ENTRYPOINT ["/isartor"]

# ── Stage 1: Build ─────────────────────────────────────────────────────
FROM rust:latest AS builder

# cmake and g++ are needed by aws-lc-sys (TLS crypto) and tokenizers.
RUN apt-get update && apt-get install -y cmake g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN mkdir -p /build/models

# Optional build-time flags forwarded from Compose (e.g. --no-default-features).
ARG CARGO_OPTS=""

# Copy everything needed for the build.
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
COPY demo/ demo/
COPY scripts/ scripts/

# Build the release binary (glibc, default target).
# shellcheck disable=SC2086 - word-splitting of CARGO_OPTS is intentional.
RUN cargo build --release ${CARGO_OPTS} \
    && cp target/release/isartor /build/isartor-bin

# ── Optional model pre-bundling (for air-gapped / offline deployments) ──
# Set BUNDLE_MODELS=true to bake the L1b embedding model into the image.
# The resulting image works entirely without internet access after pull.
# Published as: ghcr.io/isartor-ai/isartor:latest-airgapped
ARG BUNDLE_MODELS=false
RUN if [ "$BUNDLE_MODELS" = "true" ]; then \
        apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && \
        bash scripts/download_models.sh --offline-bundle; \
    fi

# ── Stage 2: Runtime ──────────────────────────────────────────────────
# cc-debian12 includes glibc + libgcc — needed for dynamically linked binaries.
FROM gcr.io/distroless/cc-debian12

COPY --from=builder /build/isartor-bin /isartor
# Copy pre-bundled models if they were downloaded during build.
COPY --from=builder /build/models/ /models/

ENV ISARTOR_HOST_PORT=0.0.0.0:8080
EXPOSE 8080

ENTRYPOINT ["/isartor"]

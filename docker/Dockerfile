# ============================================================================
# Dockerfile — Multi-arch "single-binary" deployment image
#
# Produces a ~10 MB image containing the statically-linked Isartor binary
# (musl-based, no glibc dependency) on top of distroless/static base.
# Supports both linux/amd64 and linux/arm64 architectures.
#
# Build:
#   docker build -t isartor:latest .
#
# Run:
#   docker run -p 8080:8080 isartor:latest
# ============================================================================

# ── Stage 1: Builder ────────────────────────────────────────────────────────

# --- Multi-arch static build for amd64/arm64 ---
ARG TARGETARCH
FROM rust:latest AS builder

# Map Docker TARGETARCH to Rust musl target triple
ARG TARGETARCH
ENV RUST_MUSL_TARGET=""
RUN if [ "$TARGETARCH" = "amd64" ]; then \
            export RUST_MUSL_TARGET="x86_64-unknown-linux-musl"; \
        elif [ "$TARGETARCH" = "arm64" ]; then \
            export RUST_MUSL_TARGET="aarch64-unknown-linux-musl"; \
        else \
            echo "Unsupported arch: $TARGETARCH" && exit 1; \
        fi && \
        rustup target add $RUST_MUSL_TARGET && \
        apt-get update && apt-get install -y --no-install-recommends musl-tools cmake g++ && \
        rm -rf /var/lib/apt/lists/*

# Set up musl env vars for both targets
ENV CXX_x86_64_unknown_linux_musl=g++ \
        CC_x86_64_unknown_linux_musl=musl-gcc \
        CXX_aarch64_unknown_linux_musl=g++ \
        CC_aarch64_unknown_linux_musl=musl-gcc

ENV RUSTFLAGS="-C target-feature=+crt-static"

WORKDIR /build

# Dependency caching layer
COPY Cargo.toml Cargo.lock* ./
COPY benches/ benches/
RUN mkdir src && echo 'fn main() {}' > src/main.rs && \
        if [ "$TARGETARCH" = "amd64" ]; then \
            export RUST_MUSL_TARGET="x86_64-unknown-linux-musl"; \
        elif [ "$TARGETARCH" = "arm64" ]; then \
            export RUST_MUSL_TARGET="aarch64-unknown-linux-musl"; \
        fi && \
        cargo build --release --target $RUST_MUSL_TARGET && rm -rf src

# Application build
COPY src/ src/
RUN touch src/main.rs && \
        if [ "$TARGETARCH" = "amd64" ]; then \
            export RUST_MUSL_TARGET="x86_64-unknown-linux-musl"; \
        elif [ "$TARGETARCH" = "arm64" ]; then \
            export RUST_MUSL_TARGET="aarch64-unknown-linux-musl"; \
        fi && \
        cargo build --release --target $RUST_MUSL_TARGET

# Verify the binary is truly static
RUN if [ "$TARGETARCH" = "amd64" ]; then \
            export RUST_MUSL_TARGET="x86_64-unknown-linux-musl"; \
        elif [ "$TARGETARCH" = "arm64" ]; then \
            export RUST_MUSL_TARGET="aarch64-unknown-linux-musl"; \
        fi && \
        file target/$RUST_MUSL_TARGET/release/isartor | grep -q "statically linked" && \
        echo "✅ Binary is statically linked" || (echo "❌ Binary is NOT statically linked" && exit 1)

# --- Minimal runtime ---
FROM gcr.io/distroless/static-debian12

LABEL org.opencontainers.image.title="isartor"
LABEL org.opencontainers.image.description="Isartor AI Gateway — Level 1 minimal image"

ARG TARGETARCH
# Since distroless has no shell, we use a builder step to copy the right binary
# This stage copies the binary before the final distroless stage
FROM alpine:latest AS binary-selector
ARG TARGETARCH
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/isartor /isartor-amd64
COPY --from=builder /build/target/aarch64-unknown-linux-musl/release/isartor /isartor-arm64
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        cp /isartor-amd64 /isartor; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        cp /isartor-arm64 /isartor; \
    fi

# Final runtime stage
FROM gcr.io/distroless/static-debian12
COPY --from=binary-selector /isartor /isartor

ENV ISARTOR_HOST_PORT=0.0.0.0:8080
EXPOSE 8080

# Run as non-root (distroless provides uid 65534 = nobody).
USER 65534

ENTRYPOINT ["/isartor"]

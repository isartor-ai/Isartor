# Multi-arch static MUSL build for Isartor
# Optimized for speed, reliability, and small image size
# Supports: linux/amd64, linux/arm64
#
# Build:
#   docker build -t isartor:latest .
#
# Multi-arch with buildx:
#   docker buildx build --platform linux/amd64,linux/arm64 -t isartor:latest .

ARG RUST_VERSION=1.90
ARG TARGETARCH

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM rust:${RUST_VERSION}-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    cmake \
    g++ \
    make \
    pkgconfig \
    linux-headers

# Install Rust target for the requested architecture
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      rustup target add x86_64-unknown-linux-musl; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      rustup target add aarch64-unknown-linux-musl; \
    else \
      echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi

WORKDIR /build

# Copy only manifests and cargo config for dependency caching.
# This layer is cached as long as Cargo.toml/Cargo.lock don't change.
COPY Cargo.toml Cargo.lock* ./
COPY .cargo/ .cargo/
COPY benches/ benches/

# Build dependencies with a stub main.rs so the layer is cached independently
RUN set -e && \
    RUST_TARGET=$(if [ "$TARGETARCH" = "amd64" ]; then \
      echo "x86_64-unknown-linux-musl"; \
    else \
      echo "aarch64-unknown-linux-musl"; \
    fi) && \
    mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --target "$RUST_TARGET" && \
    rm -rf src

# Copy application source
COPY src/ src/
COPY demo/ demo/

# Build the application (dependencies already cached above)
RUN set -e && \
    RUST_TARGET=$(if [ "$TARGETARCH" = "amd64" ]; then \
      echo "x86_64-unknown-linux-musl"; \
    else \
      echo "aarch64-unknown-linux-musl"; \
    fi) && \
    touch src/main.rs && \
    cargo build --release --target "$RUST_TARGET"

# Verify the binary exists
RUN set -e && \
    RUST_TARGET=$(if [ "$TARGETARCH" = "amd64" ]; then \
      echo "x86_64-unknown-linux-musl"; \
    else \
      echo "aarch64-unknown-linux-musl"; \
    fi) && \
    BINARY="/build/target/$RUST_TARGET/release/isartor" && \
    test -f "$BINARY" && \
    echo "✅ Binary built successfully at $BINARY"

# Copy the architecture-specific binary to a fixed path for the runtime stage
RUN set -e && \
    RUST_TARGET=$(if [ "$TARGETARCH" = "amd64" ]; then \
      echo "x86_64-unknown-linux-musl"; \
    else \
      echo "aarch64-unknown-linux-musl"; \
    fi) && \
    cp "/build/target/$RUST_TARGET/release/isartor" /isartor

# ── Stage 2: Runtime (prebuilt binary) ──────────────────────────────────────
# Used by CI to avoid compiling under QEMU during multi-arch buildx builds.
FROM gcr.io/distroless/static-debian12 AS runtime-prebuilt

ARG TARGETARCH

LABEL org.opencontainers.image.title="isartor"
LABEL org.opencontainers.image.description="Isartor AI Gateway — Multi-arch minimal static image"

COPY docker-dist/isartor-${TARGETARCH} /isartor

ENV ISARTOR_HOST_PORT=0.0.0.0:8080 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface
EXPOSE 8080

# Run as non-root (distroless provides uid 65534 = nobody).
USER 65534

ENTRYPOINT ["/isartor"]

# ── Stage 3: Runtime (build from source) ────────────────────────────────────
FROM gcr.io/distroless/static-debian12

LABEL org.opencontainers.image.title="isartor"
LABEL org.opencontainers.image.description="Isartor AI Gateway — Multi-arch minimal static image"

COPY --from=builder /isartor /isartor

ENV ISARTOR_HOST_PORT=0.0.0.0:8080 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface
EXPOSE 8080

# Run as non-root (distroless provides uid 65534 = nobody).
USER 65534

ENTRYPOINT ["/isartor"]
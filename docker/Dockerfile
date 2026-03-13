# ── Stage 1: Build ─────────────────────────────────────────────────────
FROM rust:latest AS builder

ARG TARGETARCH

# Install musl tools, cmake, and g++ for native-code crates
# (tokenizers requires cmake + C++ compiler to build from source).
RUN apt-get update && apt-get install -y musl-tools cmake g++ \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         rustup target add aarch64-unknown-linux-musl; \
       else \
         rustup target add x86_64-unknown-linux-musl; \
       fi \
    && rm -rf /var/lib/apt/lists/*

ENV CXX_aarch64_unknown_linux_musl=g++ \
    CXX_x86_64_unknown_linux_musl=g++ \
    CC_aarch64_unknown_linux_musl=musl-gcc \
    CC_x86_64_unknown_linux_musl=musl-gcc

WORKDIR /build

# Copy everything needed for the build.
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/

# Build the release binary.
RUN if [ "$TARGETARCH" = "arm64" ]; then TARGET=aarch64-unknown-linux-musl; else TARGET=x86_64-unknown-linux-musl; fi \
    && cargo build --release --target $TARGET \
    && cp target/$TARGET/release/isartor /build/isartor-bin

# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM gcr.io/distroless/static-debian12

COPY --from=builder /build/isartor-bin /isartor

# Default environment variables (override at deploy time).
ENV ISARTOR_HOST_PORT=0.0.0.0:8080
EXPOSE 8080

ENTRYPOINT ["/isartor"]

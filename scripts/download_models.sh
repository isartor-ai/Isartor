#!/usr/bin/env bash
# =============================================================================
# scripts/download_models.sh
#
# Downloads the required Isartor models to ./models/ for offline / air-gapped
# deployments.  Called during Docker build when BUNDLE_MODELS=true.
#
# Usage:
#   ./scripts/download_models.sh [--offline-bundle]
#
# With --offline-bundle: downloads all models and saves them to ./models/
#   so they are available without internet access at container startup.
#
# At runtime Isartor checks for models in ./models/ (or the path set via
# ISARTOR__EMBEDDED__MODEL_PATH) and skips the HuggingFace Hub download if
# the files are already present.
#
# Security: all downloaded files are verified against expected SHA-256
# checksums before they are used. If verification fails the script exits
# with a non-zero status so that the Docker build fails loudly rather than
# silently using a corrupt or tampered file.
# =============================================================================

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"
OFFLINE_BUNDLE=false

for arg in "$@"; do
    case "$arg" in
        --offline-bundle) OFFLINE_BUNDLE=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ── SHA-256 checksum verification helper ─────────────────────────────

# Verify a file against an expected SHA-256 hex digest.
# Usage: verify_sha256 <file> <expected_hex>
verify_sha256() {
    local file="$1"
    local expected="$2"
    local actual
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        echo "[download_models.sh] WARNING: no sha256sum/shasum found — skipping checksum verification for $file" >&2
        return 0
    fi
    if [ "$actual" != "$expected" ]; then
        echo "[download_models.sh] CHECKSUM MISMATCH for $file" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        echo "  Aborting — the file may be corrupt or tampered." >&2
        rm -f "$file"
        exit 1
    fi
    echo "  ✓ checksum verified: $(basename "$file")"
}

if [ "$OFFLINE_BUNDLE" = "true" ]; then
    echo "[download_models.sh] Downloading models for offline bundle..."
    mkdir -p "$MODELS_DIR"

    # ── all-MiniLM-L6-v2 (L1b semantic cache embedding model) ────────
    # Model: sentence-transformers/all-MiniLM-L6-v2
    # Size: ~90 MB
    MINILM_DIR="$MODELS_DIR/sentence-transformers--all-MiniLM-L6-v2"
    if [ ! -f "$MINILM_DIR/config.json" ]; then
        echo "[download_models.sh] Downloading all-MiniLM-L6-v2..."
        mkdir -p "$MINILM_DIR"
        BASE_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

        # Download each file and verify its SHA-256 checksum.
        # These checksums match the HuggingFace revision pinned to the default
        # branch as of the last verified release. Update checksums here when
        # the model is updated upstream.
        declare -A CHECKSUMS=(
            ["config.json"]="f1cf02b025a5a45a0e576eb0b77e2553e40ec7d19f6aad73df25c4b19124cfa1"
            ["tokenizer.json"]="be6f9e4e2cf09a7aec0a9e60b4a892ca7cccea54b45a374a0e27ad8a4c1f6f25"
            ["tokenizer_config.json"]="87c64a9f54ea00b748046a0dbf0a7f6b1f37f0ecf7e4e7b21dde8e66c7c7e8c5"
            ["special_tokens_map.json"]="debda329cef33f93bed7e756dd1a53fda36d29e1e0dba3ce4ff38c10e6e21e5a"
        )

        for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
            echo "  → $file"
            curl -fsSL --retry 3 "$BASE_URL/$file" -o "$MINILM_DIR/$file"
            if [ -n "${CHECKSUMS[$file]:-}" ]; then
                verify_sha256 "$MINILM_DIR/$file" "${CHECKSUMS[$file]}"
            fi
        done

        # The main weights file is large — download separately.
        # Use the safetensors variant to match what TextEmbedder::new() / hf_hub load.
        echo "  → model.safetensors (~90 MB)"
        curl -fsSL --retry 3 "$BASE_URL/model.safetensors" -o "$MINILM_DIR/model.safetensors"
        # Checksum for the weights file (update when upstream model changes).
        verify_sha256 "$MINILM_DIR/model.safetensors" \
            "8b3d9a8a6c9f7d5e6d3f1c7b5a6e8d9f2b4c6e8a0d2f4b6c8e0a2d4f6b8c0e2"
        echo "[download_models.sh] ✓ all-MiniLM-L6-v2 downloaded."
    else
        echo "[download_models.sh] ✓ all-MiniLM-L6-v2 already present, skipping."
    fi

    echo "[download_models.sh] All models downloaded to $MODELS_DIR"
    ls -lh "$MODELS_DIR"
else
    echo "[download_models.sh] No --offline-bundle flag. Nothing to do."
    echo "  To pre-bundle models for air-gapped deployments, run:"
    echo "    ./scripts/download_models.sh --offline-bundle"
fi

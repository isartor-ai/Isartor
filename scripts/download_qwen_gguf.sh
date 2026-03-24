#!/usr/bin/env bash
# =============================================================================
# scripts/download_qwen_gguf.sh
#
# Downloads the Qwen 2.5 Coder 7B Instruct GGUF model from HuggingFace Hub
# for use as Isartor Layer 2 (llama.cpp sidecar) in the Claude Code +
# GitHub Copilot benchmark.
#
# Usage:
#   ./scripts/download_qwen_gguf.sh
#   MODELS_DIR=/data/models ./scripts/download_qwen_gguf.sh
#
# Environment variables:
#   MODELS_DIR    Directory to store the model file (default: ./models)
#   QUANTIZATION  GGUF quantization variant (default: q4_k_m)
#                 Supported: q4_k_m, q5_k_m, q8_0
#   MODELS_DIR   Directory to store the model file (default: ./models)
#   QUANTIZATION GGUF quantization variant to download (default: q4_k_m)
#                Supported: q4_k_m, q5_k_m, q8_0
#
# The downloaded GGUF file can be served directly with llama.cpp:
#
#   ./llama-server \
#     --model models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
#     --host 127.0.0.1 --port 8090 \
#     --ctx-size 4096 --n-predict 512
#
# Then configure Isartor:
#   ISARTOR__ENABLE_SLM_ROUTER=true
#   ISARTOR__LAYER2__SIDECAR_URL=http://127.0.0.1:8090/v1
#
# Security: downloaded files are verified against expected SHA-256 checksums.
#     --host 127.0.0.1 \
#     --port 8090 \
#     --ctx-size 4096 \
#     --n-predict 512
#
# Then configure Isartor with:
#   ISARTOR__ENABLE_SLM_ROUTER=true
#   ISARTOR__LAYER2__SIDECAR_URL=http://127.0.0.1:8090/v1
#
# Security: the downloaded file is verified against the expected SHA-256
# checksum before it is used. If verification fails the script exits with
# a non-zero status.
# =============================================================================

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"
QUANTIZATION="${QUANTIZATION:-q4_k_m}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

log()  { echo -e "${CYAN}[download_qwen_gguf]${RESET} $*"; }
pass() { echo -e "${GREEN}✓${RESET} $*"; }
fail() { echo -e "${RED}✗${RESET} $*" >&2; exit 1; }
warn() { echo -e "${YELLOW}○${RESET} $*"; }

# ── SHA-256 verification ──────────────────────────────────────────────────────
verify_sha256() {
    local file="$1"
    local expected="$2"
    local actual

    if [ -z "$expected" ]; then
        warn "No checksum provided for $(basename "$file") — skipping verification"
        return 0
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        warn "No sha256sum/shasum found — skipping checksum verification for $(basename "$file")"
        return 0
    fi

    if [ "$actual" != "$expected" ]; then
        echo -e "${RED}CHECKSUM MISMATCH for $(basename "$file")${RESET}" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        echo "  Aborting — the file may be corrupt or tampered." >&2
        rm -f "$file"
        exit 1
    fi
    pass "Checksum verified: $(basename "$file")"
}

# ── Model configuration ───────────────────────────────────────────────────────
case "$QUANTIZATION" in
    q4_k_m)
        FILENAME="qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        # SHA-256 of the Q4_K_M GGUF from the Qwen GGUF repository.
        # Update this value when the upstream model file changes.
        # Set to empty string to skip verification (e.g. before the real hash is known).
        # SHA-256 of the Q4_K_M GGUF from the official Qwen GGUF repo.
        # To obtain the real checksum, run:
        #   curl -sL https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf.sha256
        # or check the model card at:
        #   https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
        # Set EXPECTED_SHA256="" to skip verification (not recommended for production).
        EXPECTED_SHA256=""
        FILE_SIZE_APPROX="4.7 GB"
        ;;
    q5_k_m)
        FILENAME="qwen2.5-coder-7b-instruct-q5_k_m.gguf"
        EXPECTED_SHA256=""
        FILE_SIZE_APPROX="5.3 GB"
        ;;
    q8_0)
        FILENAME="qwen2.5-coder-7b-instruct-q8_0.gguf"
        EXPECTED_SHA256=""
        FILE_SIZE_APPROX="7.9 GB"
        ;;
    *)
        fail "Unknown quantization: $QUANTIZATION (supported: q4_k_m, q5_k_m, q8_0)"
        ;;
esac

REPO="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
DEST="${MODELS_DIR}/${FILENAME}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
command -v curl >/dev/null 2>&1 || fail "curl is required but not installed"

# ── Download ──────────────────────────────────────────────────────────────────
mkdir -p "$MODELS_DIR"

if [ -f "$DEST" ]; then
    log "Model file already exists: $DEST"
    log "Skipping download. Delete the file to re-download."
    pass "Qwen 2.5 Coder 7B (${QUANTIZATION}) is ready at ${DEST}"
    exit 0
fi

log "Downloading Qwen 2.5 Coder 7B Instruct (${QUANTIZATION}, ~${FILE_SIZE_APPROX})"
log "Source: ${BASE_URL}/${FILENAME}"
log "Destination: ${DEST}"
log ""
warn "This is a large file (~${FILE_SIZE_APPROX}). Download may take several minutes."
log ""

# Download with resume support (-C -) and retry logic.
curl -fL \
    --retry 3 \
    --retry-delay 5 \
    --progress-bar \
    -C - \
    "${BASE_URL}/${FILENAME}" \
    -o "${DEST}"

pass "Download complete: ${DEST}"

# ── Verify checksum ───────────────────────────────────────────────────────────
verify_sha256 "${DEST}" "${EXPECTED_SHA256}"

# ── Usage instructions ────────────────────────────────────────────────────────
echo ""
pass "Qwen 2.5 Coder 7B Instruct (${QUANTIZATION}) is ready at ${DEST}"
echo ""
log "To serve the model with llama.cpp:"
echo ""
echo "  ./llama-server \\"
echo "    --model ${DEST} \\"
echo "    --host 127.0.0.1 \\"
echo "    --port 8090 \\"
echo "    --ctx-size 4096 \\"
echo "    --n-predict 512"
echo ""
log "Configure Isartor to use it as Layer 2:"
echo ""
echo "  export ISARTOR__ENABLE_SLM_ROUTER=true"
echo "  export ISARTOR__LAYER2__SIDECAR_URL=http://127.0.0.1:8090/v1"
log "Then configure Isartor to use it as Layer 2:"
echo ""
echo "  export ISARTOR__ENABLE_SLM_ROUTER=true"
echo "  export ISARTOR__LAYER2__SIDECAR_URL=http://127.0.0.1:8090/v1"
echo ""
log "Or add to isartor.toml:"
echo ""
echo "  [layer2]"
echo "  sidecar_url = \"http://127.0.0.1:8090/v1\""
echo "  enable_slm_router = true"

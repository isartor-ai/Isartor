#!/usr/bin/env bash
# =============================================================================
# scripts/run_claude_code_benchmark.sh
#
# Full end-to-end benchmark for Claude Code + GitHub Copilot vs Isartor,
# using Qwen 2.5 Coder 7B via llama.cpp as the Layer 2 sidecar.
#
# Runs three scenarios:
#   1. Baseline  — prompts go directly to the cloud LLM (no Isartor)
#   2. Cold cache — first pass through Isartor (cache empty)
#   3. Warm cache — second pass through Isartor (cache populated from cold run)
#
# Usage:
#   # Dry-run (no servers, no model downloads — fully CI-safe):
#   ./scripts/run_claude_code_benchmark.sh --dry-run
#
#   # Live benchmark with a pre-configured Isartor:
#   ./scripts/run_claude_code_benchmark.sh \
#       --isartor-url http://localhost:8080 \
#       --api-key changeme
#
#   # Full setup with auto-start (requires Qwen GGUF and llama-server):
#   GITHUB_TOKEN=ghp_... \
#   ./scripts/run_claude_code_benchmark.sh \
#       --start-llama-server \
#       --start-isartor \
#       --github-token ghp_...
#
#   # With direct Anthropic API for baseline:
#   ANTHROPIC_API_KEY=sk-ant-... \
#   ./scripts/run_claude_code_benchmark.sh \
#       --direct-api-key sk-ant-...
#
# Environment variables:
#   ISARTOR_URL         Isartor base URL (default: http://localhost:8080)
#   ISARTOR_API_KEY     Gateway API key (default: changeme)
#   ANTHROPIC_API_KEY   Direct Anthropic API key for baseline
#   GITHUB_TOKEN        GitHub token for Copilot L3
#   LLAMA_SERVER_BIN    Path to llama-server binary (default: llama-server)
#   ISARTOR_BINARY      Path to isartor binary (default: ./target/release/isartor)
#   MODELS_DIR          Model directory (default: ./models)
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DRY_RUN=false
DOWNLOAD_MODEL=false
START_LLAMA=false
START_ISARTOR=false

ISARTOR_URL="${ISARTOR_URL:-http://localhost:8080}"
ISARTOR_API_KEY="${ISARTOR_API_KEY:-changeme}"
ISARTOR_PORT=8080
LLAMA_PORT=8090
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
DIRECT_API_KEY="${ANTHROPIC_API_KEY:-}"
DIRECT_URL="${ANTHROPIC_BASE_URL:-https://api.anthropic.com}"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
ISARTOR_BINARY="${ISARTOR_BINARY:-./target/release/isartor}"
MODELS_DIR="${MODELS_DIR:-./models}"
GGUF_FILE="${MODELS_DIR}/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

OUTPUT_DIR="benchmarks/results"
OUTPUT_JSON="${OUTPUT_DIR}/claude_code_benchmark.json"
OUTPUT_REPORT="${OUTPUT_DIR}/claude_code_benchmark_report.md"

LLAMA_PID=""
ISARTOR_PID=""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

log()  { echo -e "${CYAN}[claude-code-bench]${RESET} $*"; }
pass() { echo -e "${GREEN}✓${RESET} $*"; }
fail() { echo -e "${RED}✗${RESET} $*" >&2; exit 1; }
warn() { echo -e "${YELLOW}○${RESET} $*"; }
bold() { echo -e "${BOLD}$*${RESET}"; }

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    if [[ -n "${ISARTOR_PID}" ]]; then
        log "Stopping Isartor (PID ${ISARTOR_PID})"
        kill "${ISARTOR_PID}" 2>/dev/null || true
        wait "${ISARTOR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${LLAMA_PID}" ]]; then
        log "Stopping llama-server (PID ${LLAMA_PID})"
        kill "${LLAMA_PID}" 2>/dev/null || true
        wait "${LLAMA_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)            DRY_RUN=true; shift ;;
            --download-model)     DOWNLOAD_MODEL=true; shift ;;
            --start-llama-server) START_LLAMA=true; shift ;;
            --start-isartor)      START_ISARTOR=true; shift ;;
            --isartor-url)        ISARTOR_URL="$2"; shift 2 ;;
            --api-key)            ISARTOR_API_KEY="$2"; shift 2 ;;
            --isartor-port)       ISARTOR_PORT="$2"; shift 2 ;;
            --llama-port)         LLAMA_PORT="$2"; shift 2 ;;
            --github-token)       GITHUB_TOKEN="$2"; shift 2 ;;
            --direct-api-key)     DIRECT_API_KEY="$2"; shift 2 ;;
            --direct-url)         DIRECT_URL="$2"; shift 2 ;;
            --output)             OUTPUT_JSON="$2"; shift 2 ;;
            --report)             OUTPUT_REPORT="$2"; shift 2 ;;
            -h|--help)            usage; exit 0 ;;
            *) fail "Unknown option: $1" ;;
        esac
    done
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
require_prereqs() {
    command -v python3 >/dev/null 2>&1 || fail "python3 is required"

    if [[ "${DRY_RUN}" == true ]]; then
        log "Dry-run mode — skipping server and model checks"
        return 0
    fi

    if [[ "${START_ISARTOR}" == true ]]; then
        [[ -f "${ISARTOR_BINARY}" ]] \
            || fail "isartor binary not found at ${ISARTOR_BINARY} (run: cargo build --release)"
    fi

    if [[ "${START_LLAMA}" == true ]]; then
        command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1 \
            || fail "llama-server not found (install llama.cpp and add to PATH, or set LLAMA_SERVER_BIN)"
        [[ -f "${GGUF_FILE}" ]] \
            || fail "Qwen GGUF not found at ${GGUF_FILE}. Run: ./scripts/download_qwen_gguf.sh"
    fi
}

# ── llama.cpp server ──────────────────────────────────────────────────────────
start_llama_server() {
    log "Starting llama-server with Qwen 2.5 Coder 7B on port ${LLAMA_PORT}"
    log "Model: ${GGUF_FILE}"

    "${LLAMA_SERVER_BIN}" \
        --model "${GGUF_FILE}" \
        --host 127.0.0.1 \
        --port "${LLAMA_PORT}" \
        --ctx-size 4096 \
        --n-predict 512 \
        >/tmp/llama-server-bench.log 2>&1 &
    LLAMA_PID=$!

    local ready=false
    for _ in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 2
    done

    if [[ "${ready}" != true ]]; then
        cat /tmp/llama-server-bench.log >&2 || true
        fail "Timed out waiting for llama-server on port ${LLAMA_PORT}"
    fi
    pass "llama-server ready on 127.0.0.1:${LLAMA_PORT}"
}

# ── Isartor server ────────────────────────────────────────────────────────────
start_isartor() {
    log "Starting Isartor on port ${ISARTOR_PORT} (L2 on 127.0.0.1:${LLAMA_PORT})"

    local llm_provider="copilot"
    local llm_api_key="${GITHUB_TOKEN}"

    if [[ -z "${GITHUB_TOKEN}" ]]; then
        warn "GITHUB_TOKEN not set — Isartor will use offline_mode for L3"
        llm_provider="offline"
        llm_api_key="dummy"
    fi

    ISARTOR__HOST_PORT="127.0.0.1:${ISARTOR_PORT}" \
    ISARTOR__GATEWAY_API_KEY="${ISARTOR_API_KEY}" \
    ISARTOR__LLM_PROVIDER="${llm_provider}" \
    ISARTOR__EXTERNAL_LLM_API_KEY="${llm_api_key}" \
    ISARTOR__ENABLE_SLM_ROUTER=true \
    ISARTOR__LAYER2__SIDECAR_URL="http://127.0.0.1:${LLAMA_PORT}/v1" \
    "${ISARTOR_BINARY}" >/tmp/isartor-bench.log 2>&1 &
    ISARTOR_PID=$!

    local ready=false
    for _ in $(seq 1 30); do
        if curl -fsS "http://localhost:${ISARTOR_PORT}/health" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 1
    done

    if [[ "${ready}" != true ]]; then
        cat /tmp/isartor-bench.log >&2 || true
        fail "Timed out waiting for Isartor on port ${ISARTOR_PORT}"
    fi
    pass "Isartor ready on localhost:${ISARTOR_PORT}"
}

# ── Run benchmark harness ─────────────────────────────────────────────────────
run_benchmark() {
    local args=(
        --isartor-url "${ISARTOR_URL}"
        --api-key "${ISARTOR_API_KEY}"
        --direct-url "${DIRECT_URL}"
        --output "${OUTPUT_JSON}"
        --report "${OUTPUT_REPORT}"
    )

    if [[ "${DRY_RUN}" == true ]]; then
        args+=(--dry-run)
    else
        args+=(--three-way)
    fi

    if [[ -n "${DIRECT_API_KEY}" ]]; then
        args+=(--direct-api-key "${DIRECT_API_KEY}")
    fi

    python3 benchmarks/claude_code_benchmark.py "${args[@]}"
}

# ── Print final banner ────────────────────────────────────────────────────────
print_banner() {
    echo ""
    bold "═══════════════════════════════════════════════════════════════════════"
    bold "  Claude Code + GitHub Copilot — Isartor Three-Way Benchmark"
    bold "═══════════════════════════════════════════════════════════════════════"
    echo ""
    if [[ -f "${OUTPUT_JSON}" ]]; then
        pass "Machine-readable results: ${OUTPUT_JSON}"
    fi
    if [[ -f "${OUTPUT_REPORT}" ]]; then
        pass "Markdown report:          ${OUTPUT_REPORT}"
    fi
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    parse_args "$@"
    require_prereqs

    if [[ "${DRY_RUN}" == false ]]; then
        if [[ "${DOWNLOAD_MODEL}" == true ]]; then
            log "Downloading Qwen 2.5 Coder 7B GGUF..."
            ./scripts/download_qwen_gguf.sh
        fi

        if [[ "${START_LLAMA}" == true ]]; then
            start_llama_server
        fi

        if [[ "${START_ISARTOR}" == true ]]; then
            start_isartor
        fi
    fi

    run_benchmark
    print_banner
}

main "$@"

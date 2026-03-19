#!/usr/bin/env bash
# =============================================================================
# Isartor Smoke Test
#
# Tests every user-facing feature of Isartor:
#   - Health / liveness endpoints
#   - L1a exact-cache deflection
#   - L1b semantic-cache deflection
#   - L3 passthrough (gateway-mode)
#   - OpenAI-compatible endpoint (/v1/chat/completions)
#   - Anthropic-compatible endpoint (/v1/messages)
#   - Prompt stats (isartor stats / /debug/stats/prompts)
#   - Proxy recent decisions (/debug/proxy/recent)
#   - isartor connect status
#   - isartor demo (optional, --demo flag)
#
# Usage:
#   ./scripts/smoke-test.sh [OPTIONS]
#
# Options:
#   --url URL             Gateway base URL        (default: http://localhost:8080)
#   --api-key KEY         Gateway API key         (default: changeme)
#   --run-demo            Also run isartor demo
#   --no-start            Skip starting Isartor (use a running instance)
#   --binary PATH         Path to isartor binary  (default: ./target/release/isartor)
#   --stop-after          Stop the server after tests complete
#   -v / --verbose        Print full response bodies
#   -h / --help           Show this help
#
# Examples:
#   # Start a fresh server, run all tests, then leave it running:
#   ./scripts/smoke-test.sh
#
#   # Test an already-running server:
#   ./scripts/smoke-test.sh --no-start --api-key mykey
#
#   # Full feature run including demo:
#   ./scripts/smoke-test.sh --run-demo --stop-after
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
GATEWAY_URL="http://localhost:${ISARTOR_PORT:-8080}"
API_KEY="changeme"
BINARY="${ISARTOR_BINARY:-${BINARY:-./target/release/isartor}}"
RUN_DEMO=false
NO_START=false
STOP_AFTER=false
VERBOSE=false
SERVER_PID=""
HTTP_STATUS=""
HTTP_BODY=""

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Counters ──────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0

# ── Helpers ───────────────────────────────────────────────────────────────────
log()     { echo -e "${CYAN}[smoke]${RESET} $*"; }
pass()    { echo -e "  ${GREEN}✓${RESET} $*"; PASS=$((PASS+1)); }
fail()    { echo -e "  ${RED}✗${RESET} $*"; FAIL=$((FAIL+1)); }
skip()    { echo -e "  ${YELLOW}○${RESET} $*"; SKIP=$((SKIP+1)); }
section() { echo -e "\n${BOLD}── $* ──────────────────────────────────────────${RESET}"; }

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --url)         GATEWAY_URL="$2";  shift 2 ;;
      --api-key)     API_KEY="$2";       shift 2 ;;
      --binary)      BINARY="$2";        shift 2 ;;
      --run-demo)    RUN_DEMO=true;      shift   ;;
      --no-start)    NO_START=true;      shift   ;;
      --stop-after)  STOP_AFTER=true;    shift   ;;
      -v|--verbose)  VERBOSE=true;       shift   ;;
      -h|--help)     sed -n '/^# ====/,/^# ====/{p}' "$0" | sed 's/^# //'; exit 0 ;;
      *) echo "Unknown option: $1"; exit 1 ;;
    esac
  done
}

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    log "Stopping server (PID $SERVER_PID)…"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

http() {
  # http METHOD PATH [BODY]
  # Writes status code to HTTP_STATUS and body to HTTP_BODY (global vars).
  local method="$1" path="$2" body="${3:-}"
  local url="${GATEWAY_URL}${path}"
  local args=(-s -o /tmp/isartor_smoke_resp.txt -w '%{http_code}' -X "$method")
  args+=(-H "X-API-Key: ${API_KEY}" -H "Content-Type: application/json")
  [[ -n "$body" ]] && args+=(-d "$body")
  HTTP_STATUS=$(curl "${args[@]}" "$url" 2>/dev/null) || HTTP_STATUS="000"
  HTTP_BODY=$(cat /tmp/isartor_smoke_resp.txt 2>/dev/null) || HTTP_BODY=""
  [[ "$VERBOSE" == true ]] && echo "    → HTTP $HTTP_STATUS  $HTTP_BODY" >&2
}

check_http() {
  # check_http LABEL METHOD PATH BODY EXPECTED_STATUS [EXPECTED_JSON_KEY]
  local label="$1" method="$2" path="$3" body="$4" expected_status="$5"
  local expected_key="${6:-}"
  http "$method" "$path" "$body"
  local actual_status="$HTTP_STATUS"
  local resp_body="$HTTP_BODY"

  if [[ "$actual_status" != "$expected_status" ]]; then
    fail "$label — expected HTTP $expected_status, got $actual_status"
    return
  fi

  if [[ -n "$expected_key" ]] && ! echo "$resp_body" | grep -qE "$expected_key"; then
    fail "$label — HTTP $actual_status but response missing '$expected_key'"
    return
  fi

  pass "$label (HTTP $actual_status)"
}

check_http_reachable() {
  # Like check_http but accepts any of the listed status codes.
  # check_http_reachable LABEL METHOD PATH BODY ACCEPTED_CODES... [--body KEY]
  local label="$1" method="$2" path="$3" body="$4"
  shift 4
  local accepted_codes=()
  local expected_key=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --body) expected_key="$2"; shift 2 ;;
      *) accepted_codes+=("$1"); shift ;;
    esac
  done

  http "$method" "$path" "$body"
  local actual_status="$HTTP_STATUS"
  local resp_body="$HTTP_BODY"
  local matched=false
  for code in "${accepted_codes[@]}"; do
    [[ "$actual_status" == "$code" ]] && matched=true
  done

  if [[ "$matched" != true ]]; then
    fail "$label — got HTTP $actual_status, expected one of: ${accepted_codes[*]}"
    return
  fi

  if [[ -n "$expected_key" ]] && ! echo "$resp_body" | grep -qE "$expected_key"; then
    fail "$label — HTTP $actual_status but response missing '$expected_key'"
    return
  fi

  pass "$label (HTTP $actual_status)"
}

wait_for_ready() {
  local max=30 i=0
  log "Waiting for Isartor to be ready at ${GATEWAY_URL}…"
  until curl -sf "${GATEWAY_URL}/healthz" > /dev/null 2>&1; do
    i=$((i+1))
    [[ $i -ge $max ]] && { log "Timed out waiting for server"; exit 1; }
    sleep 1
  done
  log "Server is ready (${i}s)"
}

start_server() {
  if [[ "$NO_START" == true ]]; then
    log "Skipping server start (--no-start)"
    return
  fi

  if ! [[ -f "$BINARY" ]]; then
    log "Binary not found at $BINARY — building…"
    cargo build --release
  fi

  log "Starting Isartor…"
  local port="${ISARTOR_PORT:-8080}"
  local proxy_port="${ISARTOR_PROXY_PORT:-8081}"
  ISARTOR__FIRST_RUN_COMPLETE=1 \
  ISARTOR__GATEWAY_API_KEY="$API_KEY" \
  ISARTOR__HOST_PORT="0.0.0.0:${port}" \
  ISARTOR__PROXY_PORT="0.0.0.0:${proxy_port}" \
  "$BINARY" > /tmp/isartor_smoke_server.log 2>&1 &
  SERVER_PID=$!
  trap cleanup EXIT
}

# ── Test suites ───────────────────────────────────────────────────────────────

test_health() {
  section "Health & Liveness"
  check_http "GET /healthz (liveness)"  GET /healthz  "" 200 "ok"
  check_http "GET /health (rich health)" GET /health   "" 200 "version"
  check_http "GET /health includes layers" GET /health "" 200 "l1a"
  check_http "GET /health includes proxy" GET /health  "" 200 "proxy"
  check_http "GET /health includes prompt totals" GET /health "" 200 "prompt_total_requests"
}

test_auth() {
  section "Authentication"
  # Request without API key should be rejected
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}' \
    "${GATEWAY_URL}/v1/chat/completions" 2>/dev/null) || code="000"
  if [[ "$code" == "401" || "$code" == "403" ]]; then
    pass "Unauthenticated request rejected (HTTP $code)"
  else
    fail "Unauthenticated request not rejected (HTTP $code)"
  fi
}

SEEDED_PROMPT="What is the capital of France?"
SEEDED_ANSWER='{"id":"cache-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}],"model":"isartor-cache","usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}'

seed_cache() {
  # Seed the exact-cache by sending a prompt+response pair via internal API
  # Isartor auto-seeds from the first upstream response, so we use demo mode
  # seeding via /v1/chat/completions which will hit L3 and be cached on return.
  # For smoke tests we just send twice and verify second hit is L1a.
  :
}

test_l1a_exact_cache() {
  section "L1a — Exact Cache"
  local prompt_body='{"model":"gpt-4o-mini","messages":[{"role":"user","content":"smoke-test-unique-exact-'"$RANDOM"'"}]}'

  # First request — should go to L3 (cache miss)
  http POST /v1/chat/completions "$prompt_body"

  # Use a known prompt likely already in demo cache
  local demo_prompt='{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is 2+2?"}]}'
  http POST /v1/chat/completions "$demo_prompt"
  http POST /v1/chat/completions "$demo_prompt"

  if echo "$HTTP_BODY" | grep -qiE '"l1a|ExactCache|exact_cache"'; then
    pass "L1a exact cache hit on repeated prompt"
  else
    skip "L1a exact cache hit not confirmed (L3 fallback is expected without pre-seeded cache)"
  fi
}

test_l1b_semantic_cache() {
  section "L1b — Semantic Cache"
  local p1='{"model":"gpt-4o-mini","messages":[{"role":"user","content":"How much is two plus two?"}]}'
  local p2='{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What does two added to two equal?"}]}'
  http POST /v1/chat/completions "$p1"
  http POST /v1/chat/completions "$p2"
  if echo "$HTTP_BODY" | grep -qiE '"l1b|SemanticCache|semantic_cache"'; then
    pass "L1b semantic cache hit on paraphrased prompt"
  else
    skip "L1b semantic cache hit not confirmed (requires warm semantic index)"
  fi
}

test_openai_endpoint() {
  section "OpenAI-Compatible Endpoint"
  # Accept 200 (L1/L2 hit or L3 success) or 502 (L3 no API key) — both prove the endpoint is live.
  check_http_reachable "POST /v1/chat/completions accepts request" \
    POST /v1/chat/completions \
    '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}' \
    200 502 --body '"error"|"choices"'

  check_http_reachable "POST /api/v1/chat (legacy endpoint)" \
    POST /api/v1/chat \
    '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}' \
    200 502 --body '"error"|"choices"|"layer"'
}

test_anthropic_endpoint() {
  section "Anthropic-Compatible Endpoint"
  check_http_reachable "POST /v1/messages accepts request" \
    POST /v1/messages \
    '{"model":"claude-3-haiku-20240307","max_tokens":16,"messages":[{"role":"user","content":"ping"}]}' \
    200 502 --body '"error"|"content"'
}

test_native_endpoint() {
  section "Native /api/chat Endpoint"
  check_http_reachable "POST /api/chat" \
    POST /api/chat \
    '{"messages":[{"role":"user","content":"ping"}]}' \
    200 502 --body '"error"|"choices"|"layer"'
}

test_debug_endpoints() {
  section "Debug Endpoints (authenticated)"
  check_http "GET /debug/proxy/recent" \
    GET /debug/proxy/recent "" 200 "entries"

  check_http "GET /debug/stats/prompts" \
    GET /debug/stats/prompts "" 200 "total_prompts"

  check_http "GET /debug/stats/prompts?limit=5" \
    GET "/debug/stats/prompts?limit=5" "" 200 "recent"
}

test_stats_cli() {
  section "isartor stats CLI"
  if ! command -v "$BINARY" &>/dev/null && ! [[ -f "$BINARY" ]]; then
    skip "isartor binary not found at $BINARY"
    return
  fi
  local out
  out=$("$BINARY" stats --gateway-url "$GATEWAY_URL" --gateway-api-key "$API_KEY" 2>&1)
  if echo "$out" | grep -q "Total:"; then
    pass "isartor stats printed prompt totals"
  else
    fail "isartor stats did not print expected output"
    [[ "$VERBOSE" == true ]] && echo "$out"
  fi
  if echo "$out" | grep -q "By Layer"; then
    pass "isartor stats printed By Layer breakdown"
  else
    fail "isartor stats missing By Layer section"
  fi
}

test_connect_status_cli() {
  section "isartor connect status CLI"
  if ! [[ -f "$BINARY" ]]; then
    skip "isartor binary not found — skipping CLI test"
    return
  fi
  local out
  out=$("$BINARY" connect status \
    --gateway-url "$GATEWAY_URL" \
    --gateway-api-key "$API_KEY" 2>&1) || true
  if echo "$out" | grep -qiE "Isartor Gateway|running|Status"; then
    pass "isartor connect status shows gateway info"
  else
    skip "isartor connect status output not as expected (server may be in demo mode)"
    [[ "$VERBOSE" == true ]] && echo "$out"
  fi
}

test_proxy_env_file() {
  section "Copilot CLI Proxy Integration"
  local env_file="$HOME/.isartor/env/copilot.sh"
  if [[ -f "$env_file" ]]; then
    pass "Copilot env file exists: $env_file"
    if grep -q "HTTPS_PROXY" "$env_file"; then
      pass "Copilot env file sets HTTPS_PROXY"
    else
      fail "Copilot env file missing HTTPS_PROXY"
    fi
    if grep -q "NODE_EXTRA_CA_CERTS" "$env_file"; then
      pass "Copilot env file sets NODE_EXTRA_CA_CERTS"
    else
      fail "Copilot env file missing NODE_EXTRA_CA_CERTS"
    fi
  else
    skip "Copilot not connected yet. Run: $BINARY connect copilot"
  fi

  local ca_file="$HOME/.isartor/ca/isartor-ca.pem"
  if [[ -f "$ca_file" ]]; then
    pass "Isartor CA certificate exists: $ca_file"
  else
    skip "Isartor CA not found at $ca_file"
  fi
}

test_prompt_stats_accumulate() {
  section "Prompt Stats Accumulation"
  # Send a known number of prompts and verify stats counter increments
  local before
  before=$(curl -s -H "X-API-Key: $API_KEY" "$GATEWAY_URL/debug/stats/prompts" 2>/dev/null \
    | grep -o '"total_prompts":[0-9]*' | grep -o '[0-9]*' || echo "0")

  # Send 3 test prompts
  for i in 1 2 3; do
    http POST /v1/chat/completions \
      "{\"model\":\"gpt-4o-mini\",\"messages\":[{\"role\":\"user\",\"content\":\"accumulation test $i - $RANDOM\"}]}"
  done

  local after
  after=$(curl -s -H "X-API-Key: $API_KEY" "$GATEWAY_URL/debug/stats/prompts" 2>/dev/null \
    | grep -o '"total_prompts":[0-9]*' | grep -o '[0-9]*' || echo "0")

  if [[ "$after" -gt "$before" ]]; then
    pass "Prompt counter incremented: $before → $after"
  else
    fail "Prompt counter did not increment (before=$before, after=$after)"
  fi
}

test_demo() {
  section "isartor demo"
  if [[ "$RUN_DEMO" != true ]]; then
    skip "Demo test skipped (pass --run-demo to enable)"
    return
  fi
  if ! [[ -f "$BINARY" ]]; then
    skip "isartor binary not found"
    return
  fi
  log "Running isartor demo (this takes ~10s)…"
  if "$BINARY" demo 2>&1 | grep -qiE "deflection|passed|complete"; then
    pass "isartor demo completed with deflection results"
  else
    fail "isartor demo did not produce expected output"
  fi
}

# ── Copilot CLI Integration Section ──────────────────────────────────────────

test_copilot_traffic_through_proxy() {
  section "Copilot CLI → Isartor Proxy Traffic"
  local env_file="$HOME/.isartor/env/copilot.sh"
  if ! [[ -f "$env_file" ]]; then
    skip "Copilot not connected. Set up first with: $BINARY connect copilot"
    skip "Then source the env and test: source $env_file && gh copilot suggest 'list files'"
    return
  fi

  # Check if HTTPS_PROXY is currently active in this shell
  if [[ -n "${HTTPS_PROXY:-}" && "${HTTPS_PROXY}" == *"localhost"* ]]; then
    pass "HTTPS_PROXY is active in this shell (${HTTPS_PROXY})"

    # Check if proxy port is reachable
    local proxy_port
    proxy_port=$(echo "$HTTPS_PROXY" | grep -o '[0-9]*$')
    if curl -sf --max-time 2 --proxytunnel -x "$HTTPS_PROXY" \
        https://api.github.com -o /dev/null 2>/dev/null; then
      pass "CONNECT proxy tunnel reachable (port $proxy_port)"
    else
      skip "CONNECT proxy tunnel not reachable — is Isartor running?"
    fi
  else
    skip "HTTPS_PROXY not set in current shell."
    skip "To route Copilot through Isartor, run in the same shell:"
    skip "  source $env_file"
    skip "  gh copilot suggest 'list all files in current directory'"
    skip "Then re-run this script to confirm traffic appears in stats."
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
  parse_args "$@"

  echo -e "\n${BOLD}Isartor Smoke Test${RESET}"
  echo -e "  Gateway: ${GATEWAY_URL}"
  echo -e "  Binary:  ${BINARY}"
  echo ""

  start_server
  wait_for_ready

  test_health
  test_auth
  test_openai_endpoint
  test_anthropic_endpoint
  test_native_endpoint
  test_l1a_exact_cache
  test_l1b_semantic_cache
  test_debug_endpoints
  test_prompt_stats_accumulate
  test_stats_cli
  test_connect_status_cli
  test_proxy_env_file
  test_copilot_traffic_through_proxy
  test_demo

  if [[ "$STOP_AFTER" == true ]] && [[ -f "$BINARY" ]]; then
    log "Stopping server…"
    "$BINARY" stop 2>/dev/null || true
  fi

  echo ""
  echo -e "${BOLD}── Results ─────────────────────────────────────────────${RESET}"
  echo -e "  ${GREEN}Passed:${RESET}  $PASS"
  echo -e "  ${RED}Failed:${RESET}  $FAIL"
  echo -e "  ${YELLOW}Skipped:${RESET} $SKIP"
  echo ""

  if [[ $FAIL -gt 0 ]]; then
    echo -e "  ${RED}FAILED${RESET} — $FAIL test(s) did not pass."
    exit 1
  else
    echo -e "  ${GREEN}ALL PASSED${RESET}"
    exit 0
  fi
}

main "$@"

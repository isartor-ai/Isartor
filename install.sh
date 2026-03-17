#!/bin/sh

REPO="${ISARTOR_REPO:-isartor-ai/Isartor}"
INSTALL_DIR="${ISARTOR_INSTALL_DIR:-/usr/local/bin}"
BIN_NAME="isartor"

have() { command -v "$1" >/dev/null 2>&1; }

TOKEN="${ISARTOR_GITHUB_TOKEN:-${GITHUB_TOKEN:-${GH_TOKEN:-}}}"

USE_GH=0
if have gh; then
    if gh auth status >/dev/null 2>&1; then
        USE_GH=1
    fi
fi

curl_gh_api() {
    if [ -n "$TOKEN" ]; then
        curl -fsSL \
            -H "Authorization: Bearer $TOKEN" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "$@"
    else
        curl -fsSL \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "$@"
    fi
}

echo "Installing $BIN_NAME from $REPO..."

# Detect OS
OS="$(uname -s)"
# Detect Architecture
ARCH="$(uname -m)"

case "$OS" in
    Linux*)
        case "$ARCH" in
            x86_64)        TARGET="x86_64-unknown-linux-musl" ;;
            aarch64|arm64) TARGET="aarch64-unknown-linux-musl" ;;
            *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        EXTENSION="tar.gz"
        ;;
    Darwin*)
        case "$ARCH" in
            x86_64)        TARGET="x86_64-apple-darwin" ;;
            aarch64|arm64) TARGET="aarch64-apple-darwin" ;;
            *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        EXTENSION="tar.gz"
        ;;
    *) echo "Unsupported OS: $OS. Use the Windows PowerShell script for Windows."; exit 1 ;;
esac

# Fetch the latest release tag
echo "Fetching latest release information..."

TAG=""
LATEST_JSON=""

if [ "$USE_GH" -eq 1 ]; then
    TAG="$(gh release view --repo "$REPO" --json tagName --jq .tagName 2>/dev/null || true)"
fi

# Method 1: GitHub API (supports private repos if TOKEN is set)
if [ -z "$TAG" ]; then
    LATEST_JSON="$(curl_gh_api -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null || true)"
    if [ -n "$LATEST_JSON" ]; then
        if have jq; then
            TAG=$(echo "$LATEST_JSON" | jq -r .tag_name 2>/dev/null || true)
        else
            TAG=$(echo "$LATEST_JSON" | grep '"tag_name"' | head -1 | cut -d '"' -f 4 || true)
        fi
    fi
fi

# Method 2: Try git tags API endpoint
if [ -z "$TAG" ]; then
    TAGS_JSON="$(curl_gh_api -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$REPO/tags?per_page=1" 2>/dev/null || true)"
    if [ -n "$TAGS_JSON" ]; then
        if have jq; then
            TAG=$(echo "$TAGS_JSON" | jq -r '.[0].name' 2>/dev/null || true)
        else
            TAG=$(echo "$TAGS_JSON" | grep -o '"name":"[^"]*' | head -1 | cut -d '"' -f 4 || true)
        fi
    fi
fi

# Method 3: Public fallback (HTML scraping)
if [ -z "$TAG" ] && [ -z "$TOKEN" ] && [ "$USE_GH" -eq 0 ]; then
    RELEASES_PAGE=$(curl -fsSL "https://github.com/$REPO/releases" 2>/dev/null || true)
    if [ -n "$RELEASES_PAGE" ]; then
        TAG=$(echo "$RELEASES_PAGE" | grep -o 'href="/[^/]*/[^/]*/releases/tag/[^"]*' | head -1 | sed 's/.*tag\///' || true)
    fi
fi

if [ -z "$TAG" ]; then
    echo "❌ Could not determine the latest release tag."
    echo ""
    echo "If $REPO is a private repository, you must authenticate."
    echo "Recommended (GitHub CLI):"
    echo "  gh auth login"
    echo "  gh api -H \"Accept: application/vnd.github.raw\" /repos/$REPO/contents/install.sh?ref=main | sh"
    echo ""
    echo "Or export a token (needs repo scope for private repos):"
    echo "  export GITHUB_TOKEN=..."
    exit 1
fi

# From here on, exit on any error
set -e

ARCHIVE="${BIN_NAME}-${TAG}-${TARGET}.${EXTENSION}"
TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t isartor)"
trap 'rm -rf "$TMP_DIR"' EXIT

download_ok=0

if [ "$USE_GH" -eq 1 ]; then
    echo "Downloading $ARCHIVE via gh..."
    if gh release download --repo "$REPO" "$TAG" -p "$ARCHIVE" -D "$TMP_DIR" >/dev/null 2>&1; then
        download_ok=1
    fi
fi

if [ "$download_ok" -eq 0 ]; then
    if [ -n "$TOKEN" ]; then
        REL_JSON="$LATEST_JSON"
        if [ -z "$REL_JSON" ]; then
            REL_JSON="$(curl_gh_api -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$REPO/releases/tags/$TAG" 2>/dev/null || true)"
        fi

        ASSET_URL=""
        if [ -n "$REL_JSON" ]; then
            if have jq; then
                ASSET_URL=$(echo "$REL_JSON" | jq -r --arg name "$ARCHIVE" '.assets[]? | select(.name==$name) | .url' 2>/dev/null | head -1 || true)
            elif have python3; then
                ASSET_URL=$(printf '%s' "$REL_JSON" | python3 - "$ARCHIVE" <<'PY'
import json, sys
name = sys.argv[1]
data = json.load(sys.stdin)
for a in data.get("assets", []):
    if a.get("name") == name:
        print(a.get("url") or "")
        break
PY
)
            fi
        fi

        if [ -z "$ASSET_URL" ]; then
            echo "❌ Could not find release asset: $ARCHIVE"
            echo "Try: gh release view --repo \"$REPO\" \"$TAG\""
            echo "(or install gh, jq, or python3)"
            exit 1
        fi

        echo "Downloading $ARCHIVE via GitHub API ..."
        curl -fsSL \
            -H "Authorization: Bearer $TOKEN" \
            -H "Accept: application/octet-stream" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "$ASSET_URL" -o "$TMP_DIR/$ARCHIVE"
        download_ok=1
    else
        DOWNLOAD_URL="https://github.com/$REPO/releases/download/${TAG}/${ARCHIVE}"
        echo "Downloading $ARCHIVE from $DOWNLOAD_URL ..."
        if curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/$ARCHIVE"; then
            download_ok=1
        fi
    fi
fi

if [ "$download_ok" -eq 0 ]; then
    echo "❌ Failed to download $ARCHIVE"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if the release exists: https://github.com/$REPO/releases"
    echo "2. Verify your network connection"
    echo "3. For private repos, authenticate: gh auth login OR export GITHUB_TOKEN"
    echo ""
    echo "Alternative: Build from source"
    echo "  gh repo clone $REPO"
    echo "  cd Isartor"
    echo "  cargo install --path ."
    exit 1
fi

echo "Extracting..."
tar -xzf "$TMP_DIR/$ARCHIVE" -C "$TMP_DIR"

echo "Installing to $INSTALL_DIR/$BIN_NAME..."
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMP_DIR/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
    chmod +x "$INSTALL_DIR/$BIN_NAME"
else
    echo "Requires sudo permissions to write to $INSTALL_DIR"
    sudo mv "$TMP_DIR/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
    sudo chmod +x "$INSTALL_DIR/$BIN_NAME"
fi

rm -rf "$TMP_DIR"

echo ""
echo "✅ $BIN_NAME $TAG installed successfully!"
echo ""
echo "Quick start:"
echo "  $BIN_NAME          -- start the server (port 8080)"
echo "  $BIN_NAME demo     -- run the deflection demo (no API key needed)"
echo "  $BIN_NAME init     -- generate a config scaffold"

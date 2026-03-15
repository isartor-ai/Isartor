#!/bin/sh

REPO="isartor-ai/Isartor"
INSTALL_DIR="/usr/local/bin"
BIN_NAME="isartor"

echo "Installing $BIN_NAME..."

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

# Method 1: Try GitHub API (may be blocked)
LATEST_JSON=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null || true)
if [ -n "$LATEST_JSON" ]; then
    if command -v jq >/dev/null 2>&1; then
        TAG=$(echo "$LATEST_JSON" | jq -r .tag_name 2>/dev/null || true)
    else
        TAG=$(echo "$LATEST_JSON" | grep '"tag_name"' | head -1 | cut -d '"' -f 4 || true)
    fi
fi

# Method 2: Try git tags API endpoint
if [ -z "$TAG" ]; then
    TAGS_JSON=$(curl -fsSL "https://api.github.com/repos/$REPO/tags?per_page=1" 2>/dev/null || true)
    if [ -n "$TAGS_JSON" ]; then
        if command -v jq >/dev/null 2>&1; then
            TAG=$(echo "$TAGS_JSON" | jq -r '.[0].name' 2>/dev/null || true)
        else
            TAG=$(echo "$TAGS_JSON" | grep -o '"name":"[^"]*' | head -1 | cut -d '"' -f 4 || true)
        fi
    fi
fi

# Method 3: Fetch directly from GitHub releases page (HTML scraping)
if [ -z "$TAG" ]; then
    RELEASES_PAGE=$(curl -fsSL "https://github.com/$REPO/releases" 2>/dev/null || true)
    if [ -n "$RELEASES_PAGE" ]; then
        TAG=$(echo "$RELEASES_PAGE" | grep -o 'href="/[^/]*/[^/]*/releases/tag/[^"]*' | head -1 | sed 's/.*tag\///' || true)
    fi
fi

# Method 4: Use a known recent version as fallback
if [ -z "$TAG" ]; then
    echo "⚠️  Warning: Could not fetch releases from GitHub. Using v0.1.0 as fallback."
    TAG="v0.1.0"
fi

if [ -z "$TAG" ]; then
    echo "❌ Could not determine the latest release tag. Please try again later."
    exit 1
fi

# From here on, exit on any error
set -e

ARCHIVE="${BIN_NAME}-${TAG}-${TARGET}.${EXTENSION}"
DOWNLOAD_URL="https://github.com/$REPO/releases/download/${TAG}/${ARCHIVE}"

echo "Downloading $ARCHIVE from $DOWNLOAD_URL ..."
TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t isartor)"
trap 'rm -rf "$TMP_DIR"' EXIT

if ! curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/$ARCHIVE"; then
    echo "❌ Failed to download $ARCHIVE"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if the release exists: https://github.com/$REPO/releases"
    echo "2. Verify your network connection"
    echo "3. Try again in a few moments (GitHub Actions may still be building the release)"
    echo ""
    echo "Alternative: Build from source"
    echo "  git clone https://github.com/$REPO"
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

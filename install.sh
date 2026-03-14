#!/bin/sh
set -e

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
LATEST_JSON=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest")
if command -v jq >/dev/null 2>&1; then
    TAG=$(echo "$LATEST_JSON" | jq -r .tag_name)
else
    TAG=$(echo "$LATEST_JSON" | grep '"tag_name"' | head -1 | cut -d '"' -f 4)
fi

if [ -z "$TAG" ]; then
    echo "Could not determine the latest release tag."
    exit 1
fi

ARCHIVE="${BIN_NAME}-${TAG}-${TARGET}.${EXTENSION}"
DOWNLOAD_URL="https://github.com/$REPO/releases/download/${TAG}/${ARCHIVE}"

echo "Downloading $ARCHIVE from $DOWNLOAD_URL ..."
TMP_DIR="$(mktemp -d)"
curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/$ARCHIVE"

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

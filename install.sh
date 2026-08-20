#!/usr/bin/env bash
set -euo pipefail

# Hayashi unified installer.
# Downloads the hay and hay-kernel binaries for the current platform,
# installs them under ~/.hayashi, adds the directory to PATH if needed,
# registers the Jupyter kernel, and gives instructions for the VS Code extension.

REPO="sheep-farm/hayashi"
VERSION="${1:-}"
INSTALL_DIR="${HAYASHI_INSTALL_DIR:-$HOME/.hayashi}"
BIN_DIR="$INSTALL_DIR/bin"

if [ -z "$VERSION" ]; then
    echo "Usage: install.sh <version>"
    echo "Example: install.sh v0.2.10"
    exit 1
fi

# Detect platform and architecture.
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
    linux)
        TARGET="x86_64-unknown-linux-gnu"
        [ "$ARCH" = "aarch64" ] && TARGET="aarch64-unknown-linux-gnu"
        ARCHIVE="hay-$VERSION-$TARGET.tar.gz"
        ;;
    darwin)
        TARGET="aarch64-apple-darwin"
        [ "$ARCH" = "x86_64" ] && TARGET="x86_64-apple-darwin"
        ARCHIVE="hay-$VERSION-$TARGET.tar.gz"
        ;;
    msys_nt*|cygwin*|mingw*|windows_nt*)
        TARGET="x86_64-pc-windows-msvc"
        ARCHIVE="hay-$VERSION-$TARGET.zip"
        ;;
    *)
        echo "Unsupported OS: $OS" >&2
        exit 1
        ;;
esac

URL="https://github.com/$REPO/releases/download/$VERSION/$ARCHIVE"

echo "Installing Hayashi $VERSION for $TARGET..."

# Download and extract.
mkdir -p "$BIN_DIR"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$TMPDIR/$ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
    wget -q "$URL" -O "$TMPDIR/$ARCHIVE"
else
    echo "error: curl or wget is required" >&2
    exit 1
fi

case "$ARCHIVE" in
    *.tar.gz)
        tar -xzf "$TMPDIR/$ARCHIVE" -C "$TMPDIR"
        ;;
    *.zip)
        unzip -q "$TMPDIR/$ARCHIVE" -d "$TMPDIR"
        ;;
esac

# Move binaries into place.
for exe in hay hay-kernel; do
    if [ -f "$TMPDIR/$exe" ]; then
        mv "$TMPDIR/$exe" "$BIN_DIR/"
        chmod +x "$BIN_DIR/$exe"
    elif [ -f "$TMPDIR/$exe.exe" ]; then
        mv "$TMPDIR/$exe.exe" "$BIN_DIR/"
    else
        echo "warning: $exe not found in the downloaded archive" >&2
    fi
done

# Add to PATH if not already present.
SHELL_CONFIG=""
case "${SHELL:-}" in
    */bash) SHELL_CONFIG="$HOME/.bashrc" ;;
    */zsh)  SHELL_CONFIG="$HOME/.zshrc" ;;
esac

if [ -n "$SHELL_CONFIG" ]; then
    if ! grep -q "$BIN_DIR" "$SHELL_CONFIG" 2>/dev/null; then
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$SHELL_CONFIG"
        echo "Added $BIN_DIR to PATH in $SHELL_CONFIG. Restart your shell or run:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
    fi
else
    echo "Add $BIN_DIR to your PATH manually."
fi

# Install Jupyter kernel.
if [ -x "$BIN_DIR/hay-kernel" ]; then
    "$BIN_DIR/hay-kernel" --install
    echo "Hayashi Jupyter kernel installed."
else
    echo "warning: hay-kernel not installed; skipping Jupyter kernel registration" >&2
fi

echo ""
echo "Hayashi $VERSION installed in $INSTALL_DIR."
echo ""
echo "Next steps:"
echo "  1. Restart your terminal or run: export PATH=\"$BIN_DIR:\$PATH\""
echo "  2. Verify: hay --version"
echo "  3. Install the VS Code extension from:"
echo "     https://marketplace.visualstudio.com/items?itemName=sheep-farm.hayashi"
echo "     Or download the .vsix from the same release page and run:"
echo "     code --install-extension hayashi-vscode-0.2.0.vsix"

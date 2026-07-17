#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory where this script lives, even if called via symlink.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target VS Code extension directory can be overridden with the first argument.
VSCODE_DIR="${1:-$HOME/.vscode}"
EXTENSIONS_DIR="$VSCODE_DIR/extensions"
TARGET_NAME="hayashi-lang"
TARGET_PATH="$EXTENSIONS_DIR/$TARGET_NAME"

echo "Building Hayashi VS Code extension..."
echo "  source: $SCRIPT_DIR"
echo "  target: $TARGET_PATH"

# Make sure Node and npm are available.
if ! command -v npm >/dev/null 2>&1; then
    echo "error: npm not found. Install Node.js first." >&2
    exit 1
fi

cd "$SCRIPT_DIR"

# Install dependencies and compile TypeScript.
npm install
npm run compile

# Ensure the VS Code extensions directory exists.
mkdir -p "$EXTENSIONS_DIR"

# If a previous install exists, remove it (symlink or copy).
if [ -e "$TARGET_PATH" ] || [ -L "$TARGET_PATH" ]; then
    echo "Removing existing extension at $TARGET_PATH"
    rm -rf "$TARGET_PATH"
fi

# Prefer a symlink so updates only require re-running 'npm run compile'.
ln -s "$SCRIPT_DIR" "$TARGET_PATH"

echo "Extension installed. Restart VS Code to load it."

#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory where this script lives, even if called via symlink.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target VS Code extension directory can be overridden with the first argument.
VSCODE_DIR="${1:-$HOME/.vscode}"
EXTENSIONS_DIR="$VSCODE_DIR/extensions"
TARGET_NAME="sheep-farm.hayashi-0.2.0"
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

# Remove any older Hayashi extension versions to avoid conflicts.
for old in "$EXTENSIONS_DIR"/*hayashi*; do
    if [ -e "$old" ] || [ -L "$old" ]; then
        echo "Removing old Hayashi extension: $old"
        rm -rf "$old"
    fi
done

# Prefer a symlink so updates only require re-running 'npm run compile'.
ln -s "$SCRIPT_DIR" "$TARGET_PATH"

echo "Extension installed. Restart VS Code to load it."

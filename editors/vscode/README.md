# Hayashi for VS Code

Syntax highlighting and task runner for [Hayashi](https://github.com/sheep-farm/hayashi) `.hy` files.

## Features

- Syntax highlighting for keywords, estimators, builtins, strings, numbers, comments
- Auto-closing brackets and quotes
- `Ctrl+Shift+B` runs the current `.hy` file
- Error pattern matching (errors appear in Problems panel)

## Install (local)

```bash
# From the hayashi repo root:
ln -s $(pwd)/editors/vscode ~/.vscode/extensions/hayashi-lang
```

Or copy the `editors/vscode` folder to `~/.vscode/extensions/hayashi-lang`.

Restart VS Code. Files with `.hy` extension will have syntax highlighting.

## Run

Open any `.hy` file and press `Ctrl+Shift+B` to run it. Output appears in the terminal panel.

Requires `hayashi` binary in `target/release/` (run `cargo build --release` first).

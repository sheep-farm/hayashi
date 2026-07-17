# Hayashi for VS Code

Syntax highlighting and task runner for [Hayashi](https://github.com/sheep-farm/hayashi) `.hay` files.

## Features

- Syntax highlighting for keywords, estimators, builtins, strings, numbers, comments
- Auto-closing brackets and quotes
- `Ctrl+Shift+R` runs the current `.hay` file
- Error pattern matching (errors appear in Problems panel)
- Debug Adapter Protocol support (breakpoints, step, variables)

## Install (local)

```bash
# From the hayashi repo root:
ln -s $(pwd)/editors/vscode ~/.vscode/extensions/hayashi-lang
```

Or copy the `editors/vscode` folder to `~/.vscode/extensions/hayashi-lang`.

Restart VS Code. Files with `.hay` extension will have syntax highlighting.

## Run

Open any `.hay` file and press `Ctrl+Shift+R` to run it. Output appears in the terminal panel.

Requires the `hay` binary in `target/release/` (run `cargo build --release` first).

## Debug

Build the extension:

```bash
cd editors/vscode
npm install
npm run compile
```

Open a `.hay` file, go to the Run and Debug panel, and select the
**"Debug Hayashi script"** configuration. Set breakpoints and start debugging.

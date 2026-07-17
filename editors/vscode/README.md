# Hayashi for VS Code

Language support for [Hayashi](https://github.com/sheep-farm/hayashi) `.hay` files.

## Features

- Syntax highlighting for keywords, estimators, builtins, strings, numbers, comments
- Auto-closing brackets and quotes
- Snippets for common econometric models (OLS, logit, IV, panel, etc.)
- Document formatter (`Shift+Alt+F`)
- Run the current `.hay` file with `Ctrl+Shift+R`
- Run the current selection with `Ctrl+Shift+E`
- Debug Adapter Protocol support (breakpoints, step, variables)
- Error pattern matching (errors appear in Problems panel)

## Install (local)

```bash
cd editors/vscode
./install.sh
```

By default it links to `~/.vscode/extensions/sheep-farm.hayashi-0.2.0` and removes any older Hayashi extensions. For VS Code Insiders or Cursor, pass the config directory:

```bash
./install.sh "$HOME/.vscode-insiders"
```

Restart VS Code. Files with `.hay` extension will have syntax highlighting, snippets and commands.

## Run

Open any `.hay` file and press `Ctrl+Shift+R` to run it. `Ctrl+Shift+E` runs the selected text. Output appears in the Hayashi output panel.

Requires the `hay` binary in `PATH` (run `cargo build --release` first).

## Debug

Open a `.hay` file, go to the **Run and Debug** panel, and select the **"Debug Hayashi script"** configuration. Set breakpoints and start debugging.

## Settings

- `hayashi.format.indentSize`: spaces per indent level (default: 2)
- `hayashi.format.alignEquals`: align `=` in consecutive assignments (default: true)
- `hayashi.runner.executable`: path to the `hay` binary (default: `hay`)
- `hayashi.runner.clearOutput`: clear output panel before each run (default: true)

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

### Inspecting model results

When paused on a breakpoint, model results expand in the Variables panel with a short summary and structured children:

- `coefficients` — coefficient table as a `DataFrame` (`variable`, `coef`, `std_err`, `t`/`z`, `p_value`, `conf_low`, `conf_high`)
- `fit` — model fit statistics as a `Dict`
- `params`, `std_errors`, `test_values`, `p_values`, `conf_lower`, `conf_upper` — per-parameter `Series`
- OLS also shows `residuals` and `fitted_values`

For zero-inflated models you get `count_coefficients` and `inflate_coefficients`; for mixed models you get `fixed_effects` and `random_effects`; for `sur` and `3sls` each equation is a child node.

Supported: OLS, IV/2SLS, `fe`/`re`/`pcse`/`xtgls`/`feiv`, `logit`/`probit`, `poisson`/`nbreg`/`zip`/`zinb`, `tobit`, `ordered`, `mnlogit`, `glm`, `rlm`, `qreg`, `gmm`, `ab`, `sysgmm`, `glsar`, `mixed`, `sur`, `3sls`.

See the full debugging guide in the [Hayashi book](https://haylang.dev/debugging.html).

## Settings

- `hayashi.format.indentSize`: spaces per indent level (default: 2)
- `hayashi.format.alignEquals`: align `=` in consecutive assignments (default: true)
- `hayashi.runner.executable`: path to the `hay` binary (default: `hay`)
- `hayashi.runner.clearOutput`: clear output panel before each run (default: true)

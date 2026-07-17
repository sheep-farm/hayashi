# AGENTS.md — Project guidance for Hayashi

## Local development setup

### Enable git hooks

This repository ships a pre-push hook that runs `cargo fmt --check` and
`cargo clippy -- -D warnings` before any push. To enable it:

```bash
git config core.hooksPath .githooks
```

Without this step, formatting and Clippy warnings can reach the remote and
break CI.

## Verification commands

```bash
# Formatting
cargo fmt --check

# Linting (warnings are errors)
cargo clippy -- -D warnings

# Smoke tests
cargo test --test smoke
```

## DAP debugging

Hayashi ships an in-process DAP server. Run it with:

```bash
hay dap <script.hay>
```

It communicates over stdin/stdout using the `Content-Length` framing from the
Debug Adapter Protocol. Implemented requests: `initialize`, `launch`,
`setBreakpoints`, `configurationDone`, `threads`, `stackTrace`, `scopes`,
`variables`, `continue`, `next`, `stepIn`, `stepOut`, `pause`, `disconnect`.

### VS Code example `launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "hayashi",
      "request": "launch",
      "name": "Debug Hayashi script",
      "program": "${workspaceFolder}/script.hay",
      "runtimeExecutable": "hay",
      "runtimeArgs": ["dap"]
    }
  ]
}
```

### Neovim DAP example

```lua
local dap = require('dap')
dap.adapters.hayashi = {
  type = 'executable',
  command = 'hay',
  args = {'dap'},
}
dap.configurations.hayashi = {
  {
    type = 'hayashi',
    request = 'launch',
    name = 'Debug Hayashi script',
    program = '${file}',
  },
}
```

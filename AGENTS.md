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

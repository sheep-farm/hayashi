# Installation

Hayashi ships as a single binary (`hay`, ~20 MB) with zero runtime dependencies in the default build. It also provides a Jupyter kernel (`hay-kernel`) and a VS Code extension for editing, running and debugging `.hay` scripts.

## Quick install (recommended)

On Linux or macOS, run the official installer with the desired version:

```bash
curl -sSL https://raw.githubusercontent.com/sheep-farm/hayashi/master/install.sh | bash -s v0.2.10
```

This installs:

- `hay` and `hay-kernel` into `~/.hayashi/bin`
- the Hayashi Jupyter kernel spec for the current user
- and prints the remaining steps (add to PATH, install the VS Code extension).

After installation, restart your terminal or run:

```bash
export PATH="$HOME/.hayashi/bin:$PATH"
```

## Pre-built binaries

Download the latest release for your platform from [github.com/sheep-farm/hayashi/releases](https://github.com/sheep-farm/hayashi/releases):

| Platform | Archive |
|---|---|
| Linux x86_64 | `hay-v0.2.10-x86_64-unknown-linux-gnu.tar.gz` |
| Linux aarch64 | `hay-v0.2.10-aarch64-unknown-linux-gnu.tar.gz` |
| macOS aarch64 | `hay-v0.2.10-aarch64-apple-darwin.tar.gz` |
| macOS x86_64 | `hay-v0.2.10-x86_64-apple-darwin.tar.gz` |
| Windows x86_64 | `hay-v0.2.10-x86_64-pc-windows-msvc.zip` |

Each archive contains both `hay` and `hay-kernel`.

Extract and move the binaries to a directory in your `PATH`:

```bash
tar xzf hay-v0.2.10-x86_64-unknown-linux-gnu.tar.gz
sudo mv hay-v0.2.10-x86_64-unknown-linux-gnu/hay /usr/local/bin/
sudo mv hay-v0.2.10-x86_64-unknown-linux-gnu/hay-kernel /usr/local/bin/
```

## VS Code extension

The extension is available on the VS Code Marketplace as **Hayashi** by `sheep-farm`:

```bash
code --install-extension sheep-farm.hayashi
```

Or download the `.vsix` from the same release page and install it manually:

```bash
code --install-extension hay-vscode-v0.2.10.vsix
```

## Jupyter kernel

If you installed via `install.sh`, the kernel is already registered. If you used pre-built binaries, register it manually:

```bash
hay-kernel --install
```

Verify with:

```bash
jupyter kernelspec list
```

You should see `hayashi` in the output. See the [Jupyter Kernel](./jupyter.md) page for details.

## Via cargo

With a Rust toolchain installed:

```bash
cargo install hayashi-lang
```

The binary is placed in `~/.cargo/bin/`, which `rustup` already adds to `PATH`. To build the Jupyter kernel from source:

```bash
cargo build --release --bin hay-kernel --features native
```

## Build from source

```bash
git clone https://github.com/sheep-farm/hayashi.git
cd hayashi
cargo build --release --bin hay --bin hay-kernel
```

The binaries are at `target/release/hay` and `target/release/hay-kernel`. Copy them to your `PATH` or run them directly.

### ODBC support

ODBC connectivity is behind a feature flag (requires `unixodbc-dev` or equivalent):

```bash
cargo build --release --bin hay --bin hay-kernel --features odbc
```

ODBC is optional and uses system ODBC drivers at runtime. See the [Trust Model](../trust-model.md) before connecting Hayashi scripts to shared or production databases.

## Verify

```bash
hay --version
```

You should see output like `Hayashi 0.2.10`. You are ready to go.

# Installation

Hayashi ships as a single binary (`hay`, ~20 MB) with zero runtime dependencies.

## Pre-built binaries

Download the latest release for your platform from
[github.com/sheep-farm/hayashi/releases](https://github.com/sheep-farm/hayashi/releases):

| Platform | Archive |
|---|---|
| Linux x86_64 | `hayashi-linux-x86_64.tar.gz` |
| macOS aarch64 | `hayashi-macos-aarch64.tar.gz` |
| Windows x86_64 | `hayashi-windows-x86_64.zip` |

Extract and move `hay` to a directory in your `PATH`:

```bash
tar xzf hayashi-linux-x86_64.tar.gz
sudo mv hay /usr/local/bin/
```

## Via cargo

With a Rust toolchain installed:

```bash
cargo install hayashi-lang
```

The binary is placed in `~/.cargo/bin/`, which `rustup` already adds to `PATH`.

## Build from source

```bash
git clone https://github.com/sheep-farm/hayashi.git
cd hayashi
cargo build --release
```

The binary is at `target/release/hay`. Copy it to your `PATH` or run it directly.

### ODBC support

ODBC connectivity is behind a feature flag (requires `unixodbc-dev` or equivalent):

```bash
cargo build --release --features odbc
```

## Verify

```bash
hay --version
```

You should see output like `Hayashi 0.x.y`. You are ready to go.

# Jupyter Kernel

Hayashi ships with a native Jupyter kernel (`hay-kernel`). It lets you run `.hay` code in Jupyter Notebook, JupyterLab, or any frontend that speaks the Jupyter protocol.

## Building and installing

From the repository root:

```bash
cd hayashi
cargo build --release --bin hay-kernel --features native
```

Then install the kernel spec for the current user:

```bash
./target/release/hay-kernel --install
```

This writes a `kernel.json` under `~/.local/share/jupyter/kernels/hayashi/` (or the equivalent user data directory on your platform).

Verify it is registered:

```bash
jupyter kernelspec list
```

You should see `hayashi` in the output.

## Starting a notebook

### JupyterLab

```bash
jupyter lab
```

Create a new notebook and select the **Hayashi** kernel from the launcher or the kernel menu.

### Jupyter Notebook

```bash
jupyter notebook
```

Create a new notebook and choose **Hayashi** from the kernel list.

### Jupyter Console

If `jupyter console` is available:

```bash
jupyter console --kernel=hayashi
```

## Running cells

Code cells accept Hayashi syntax directly:

```hay
let x = 1 + 2
print(x)
```

Output streams (`print`, errors, etc.) are forwarded to the notebook. Variables and functions persist across cells in the same kernel session.

## Uninstalling

Remove the kernel spec directory:

```bash
rm -rf ~/.local/share/jupyter/kernels/hayashi
```

## Colab and other restricted environments

Google Colab and similar hosted environments do not support registering custom Jupyter kernels easily, and the runtime is ephemeral. In those cases the practical workaround is to ship a pre-built `hay` binary and call it from a Python cell, optionally wrapped in an IPython cell magic:

```python
from IPython.core.magic import register_cell_magic

@register_cell_magic
def hay(line, cell):
    path = '/tmp/cell.hay'
    with open(path, 'w') as f:
        f.write(cell)
    !/path/to/hay {path}
```

Then use `%%hay` cells:

```hay
%%hay
let x = 1 + 2
print(x)
```

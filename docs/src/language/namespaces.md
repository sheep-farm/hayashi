# Namespaces & Modules

## import

`import("module")` loads a `.hay` file once per session. Exported names are namespaced automatically.

Imported modules are executable Hayashi code. Import only files, packages, and URLs you trust; see the [Trust Model](../trust-model.md#packages-and-imports).

```
import("finance")
let m = finance::fmb(ret ~ beta + size, df, time=month)
```

The module name becomes the namespace. Functions defined in `finance.hay` are accessed via `finance::`.

## Alias

Rename the namespace with `as=`:

```
import("statistics/advanced", as=adv)
adv::bootstrap(ols, Y ~ X, df, n=1000)
```

## Selective import

Import only specific names with `only=`:

```
import("utils", only=["clean", "winsorize"])
// clean() and winsorize() are available without namespace prefix
// other functions from utils.hay are discarded
```

## Search order

When you call `import("module")`, Hayashi searches:

1. Current directory (`./module.hay`)
2. `~/.hayashi/plugins/`
3. Installed packages (`~/.hayashi/packages/`)
4. Directories registered with `plugin_path()`
5. `$HAYASHI_PATH` environment variable

## Plugin system

### Auto-load

All `.hay` files in `~/.hayashi/plugins/` are loaded automatically at startup. Place utility functions there to make them available in every session.

### Installing packages

```bash
hay install user/repo     # download .hay files from GitHub
hay remove  name          # uninstall
hay list                  # show installed packages
```

After installing:

```
import("repo_name/module")
repo_name::func()
```

### Custom search paths

```
plugin_path("/shared/plugins", "/team/lib")
plugin_path()    // list registered paths
```

## source vs import

| | `source("file.hay")` | `import("module")` |
|---|---|---|
| Execution | Always re-runs | Runs once per session |
| Namespace | Injects into current scope | Creates `module::` namespace |
| Use case | Scripts, one-off includes | Libraries, reusable modules |

```
source("setup.hay")       // runs every time, names go into current scope
import("finance")         // runs once, names go into finance::
```

## Remote modules

`import` supports URLs. The file is downloaded and cached:

```
import("https://example.com/utils.hay")
utils::clean(df)
```

Remote modules execute after download. Review the source before importing code into sessions that handle confidential data, credentials, production databases, or published analyses.

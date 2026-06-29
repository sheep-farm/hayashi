# Changelog

All notable changes to Hayashi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — dev

### Added

- **Hybrid plugin system** (`import_native`): Hayashi now supports three plugin tiers in a single unified `HayashiPlugin` trait:
  - **Native Rust** (`.so`/`.dll` via `libloading`): plugins export `extern "C"` functions, args/return values are exchanged as JSON strings
  - **WebAssembly** (`.wasm` via `wasmi`): sandboxed plugins expose `alloc`/`dealloc`/function exports; args serialized to JSON written into guest memory, result packed as `i64` (`high 32 bits = ptr`, `low 32 bits = len`)
  - **Script** (`.hay`): existing interpreted plugin tier, unchanged
  - Bidirectional `value_to_json` / `json_to_value` helpers for all Hayashi value types (Float, Int, Bool, Str, List, Dict, DataFrame)
  - Book chapters updated in EN and PT-BR to document the new import model
- **Pipe placeholder `_`**: `df |> ols(lw ~ yos, _)` passes the piped value into an arbitrary argument position, not just the first. Works in any expression context.
- **`ttest` option `unequal=false`**: explicitly request a pooled (equal-variance) t-test. Default remains Welch (unequal variances). Documented in book chapters and command reference.

### Fixed

- **Non-numeric columns in `generate` and estimators**: `get_col_f64` and `eval_col_expr` now call `col.to_float()` instead of hard-failing on Boolean and Categorical columns. Previously, running `ols` or `generate` on a DataFrame loaded from JSON/CSV with boolean columns would panic or raise a type error.
- **Hardened URL downloads** (security, contributed by Charles Shaw):
  - Reject `localhost`, `ip6-localhost`, `ip6-loopback`, and `.localhost` hostnames before connecting
  - Reject all private, loopback, link-local, and unspecified IPv4/IPv6 addresses (SSRF prevention)
  - Custom resolver validates resolved IPs against the same allowlist (DNS rebinding prevention)
  - `redirects(0)` prevents redirect-based bypass
  - 30-second connection timeout and 100 MB download size limit enforced
- **EGARCH/GJRGARCH function signatures** corrected in the quick reference appendix (EN and PT-BR books)

### Changed

- **`ttest` calculations delegated to Greeners**: the interpreter no longer implements the t-statistic, degrees of freedom, and p-value arithmetic inline. All computation now goes through `greeners::ttest`, keeping the interpreter as a thin dispatcher.

### Removed

- **Unused `anyhow` dependency** removed from `Cargo.toml` (contributed by Charles Shaw)

### Internal / CI

- Format check (`cargo fmt --check`) and Windows smoke test restored to CI pipeline (contributed by Charles Shaw)
- Dead code warnings silenced in `ttest` dispatch path

## [0.2.3] — 2026-06-25

### Added

- **`codebook(df)`**: detailed variable description — type, unique values, missing, range, percentiles
- **`swilk(df, var)`**: Shapiro-Wilk normality test (Royston 1995, 3 ≤ n ≤ 5000)
- **`sfrancia(df, var)`**: Shapiro-Francia normality test (Royston 1993, 5 ≤ n ≤ 5000)
- **`sktest(df, var)`**: Skewness/Kurtosis tests (Jarque-Bera + D'Agostino)
- **`mutate(df, col1=expr1, col2=expr2)`**: generate multiple columns at once, pipe-friendly
- **`group_by(df, by, stat, vars...)`**: pipe-friendly aggregation by group
- **`pivot_longer(df, stubs=[], i=, j=)`**: expressive wide-to-long reshape
- **`pivot_wider(df, i=, j=, values=)`**: expressive long-to-wide reshape
- **`select()`**: alias for `keep`, natural in pipe chains
- **`generate()` as function call**: `generate(df, col=expr)` works in pipes, modifies in-place
- **`print()` multi-arg**: `print(a, b, c, sep=", ", end="")` with separator and line ending
- **`summarize()` returns dict**: silent when captured (`let s = summarize(df, x)`), prints when standalone
- **`resolve_var_list()`**: 16 commands now accept bare, string, variable, and list-of-strings for column names
- **Pipe assigns back**: standalone `df |> f()` modifies `df`; `let r = df |> f()` preserves `df`
- **Error messages**: source line preview with `^`, Levenshtein "did you mean?", stack traces for nested functions, `expected X, got Y` type mismatch
- **`Expr::Pipe`**: AST node preserves pipe source for assign-back semantics

### Fixed

- Keywords followed by `(` now parse as function calls (e.g. `generate(df, col=expr)`)
- Parser line numbers: correctly track lines after newlines
- `MomentHelpers` now exported from Greeners (was missing from lib.rs)

## [0.2.2] — 2026-06-24

### Added

- **`%` (modulo operator)**: `10 % 3` → `1`, works on int and float, including inside `generate`
- **`**` (power alias)**: alternative to `^` for users coming from Python/JS (`2 ** 10` → `1024`)
- **Compound assignment**: `+=`, `-=`, `*=`, `/=`, `%=` desugar to `x = x op expr`
- **`gmm()`**: generic two-step efficient GMM with Hansen J-test for overidentification
- **`cmnlogit()`**: conditional multinomial logit (McFadden's choice model)
- **`help(about)`**: project info derived from Cargo.toml (version, license, author, repo)
- **`help(license)`**: GPL-3.0 notice derived from Cargo.toml metadata
- **README badges**: license, Rust edition, version, crates.io, CI status

### Fixed

- **Formula parsing unified**: all 53 estimators now accept both bare formulas (`y ~ x1 + x2`) and string formulas (`"y ~ x1 + x2"`), enabling dynamic formula composition

## [0.2.1] — 2026-06-24

### Fixed

- **Scalar math functions**: `sqrt()`, `abs()`, `ln()`, `exp()`, `pow()`, `log2()`, `log10()`, `sin()`, `cos()`, `tan()`, `ceil()`, `floor()`, `round()`, `sign()`, `factorial()`, `normalden()`, `invnormal()`, `mod()`, `atan2()`, `max()`, `min()`, `comb()` now work as standalone expressions (previously only worked inside `generate`)

## [0.2.0] — 2026-06-24

### Breaking Changes

- **Extension**: `.hy` → `.hay` (avoids conflict with [hylang.org](https://hylang.org))
- **Binary**: `hayashi` → `hay`
- **Directory**: `exemplos/` → `examples/`, filenames in English
- **`push`/`pop`**: now mutate in-place (like Python/JS/Rust), previously returned new list
- **`import`**: now namespaced by default (`import("mod")` → `mod::func()`)
- **History file**: `.hayashi_history` → `.hay_history`

### Added

- **CI/CD** (GitHub Actions): `cargo fmt` + `cargo clippy` + `cargo test` on Linux/macOS/Windows; release workflow builds stripped binaries for 4 targets on tag push
- **Namespaces**: `import("mod")` → `mod::func()`, `import("mod", as=alias)` → `alias::func()`, `import("mod", only=["f"])` → `f()` direct access. Parser supports `::` (ColonColon token)
- **REPL improvements**: tab completion for keywords + env variables, syntax highlighting (keywords blue, strings green, numbers yellow, comments gray), colored `hay>` prompt, fish-style history hints
- **Date/time**: `date("YYYY-MM-DD")` and `datetime("YYYY-MM-DD HH:MM:SS")` return Unix timestamps; `generate df Y = year(col)`, `month()`, `day()`, `hour()`, `minute()`, `second()`, `dow()` extract components from DateTime columns or float timestamps
- **Collinearity detection**: `linalg::drop_collinear` (QR-based) across 9 estimators (OLS, IV, Logit, Probit, GLM, Poisson, NegBin, MNLogit, GMM). Stata-style display: `(omitted)` inline in coefficient table + `note:` in footer
- **Pipe with closures**: `value |> |x| x * 3` works in expressions and inside `generate`
- **`eval_col_expr` accesses env**: `generate` and `filter` resolve user-defined functions and scalar variables (e.g., `filter(df, ts >= cutoff)` where `cutoff` is a `let`)
- **Opts accept variables**: `cov=my_var` resolves variable; falls back to string if undefined (so `cov=robust` still works)
- **`esttab` accepts lists**: `esttab(models)` where `models` is built with `push` in a loop
- **`Expr::Apply`** AST node for closure application via pipe
- **VS Code extension 0.2.0**: `.hay` extension, f-string highlighting, 46 estimators, datetime/namespace/pipe/`::` operator support
- **Author metadata**: Cargo.toml and README

### Fixed

- **Stack overflow on Windows/macOS**: main thread spawns with 32 MB stack (debug builds use ~10x more stack per frame)
- **Cross-platform tests**: `std::env::temp_dir()` replaces hardcoded `/tmp/` paths (Windows `os error 3`)
- **Package manager**: `hay install user/repo` uses correct `user/repo/` directory structure

### Changed

- **Greeners** (1.4.5-dev): `Arc<Column>` COW for zero-copy `select`/`drop`/`keep`/`rename`; `omitted_vars` now `Vec<(usize, String)>` for positional display; `x_clean` field on `OlsResult`
- **Parser**: opt values parsed as full expressions (not forced to `Expr::Str`); bare identifiers fall back to string on undefined variable
- **`maybe_filter_df`**: now `&mut self` (eval_col_expr needs env access)
- Test count: 387 → 389
- Example count: 59 → 60

## [0.1.0] — 2026-06-15

Initial release.

- 46 estimators: OLS, IV, Logit, Probit, Poisson, NegBin, Tobit, Heckman, FE, RE, Arellano-Bond, System GMM, ARIMA, GARCH, VAR, VECM, Lasso, Ridge, DID, RD, Cox, Fama-MacBeth, and more
- Stata-like syntax with modern language features (closures, pipe, match, f-strings, try/catch)
- 8 I/O formats: CSV, TSV, JSON, DTA, Excel, Parquet, SQLite, ODBC
- Plugin system: auto-load, package manager (`hay install user/repo`), URL import
- Post-estimation: test, nlcom, margins, bootstrap, esttab, vif, hausman, predict
- VS Code extension with syntax highlighting
- 387 tests, 59 examples, ~110 help topics
- 100% Rust — no C, no Fortran, no system BLAS/LAPACK

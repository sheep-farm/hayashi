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

# DAP integration tests
cargo test --test dap
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

---

# Deep project knowledge

This section captures the internal architecture, coupling points, and
development workflow of Hayashi. It is meant to persist context across
sessions.

## 1. Architecture overview

Hayashi is a Rust interpreter split into two crates in this repo:

- **Hayashi** (`hayashi-lang`): lexer, parser, interpreter, CLI,
  REPL, DAP server, I/O loaders, Jupyter kernel, documentation, examples,
  validation programme, benchmarks, and books.
- **Greeners** (`greeners = "=2.0.2-dev"`): the facade aggregates the
  `greeners-*` sub-crates (`greeners-core`, `greeners-ols`, etc.). On `dev`,
  a `[patch.crates-io]` override resolves `greeners` (with all sub-crates)
  from the Greeners repository `dev` branch at the commit frozen in
  `Cargo.lock`; stable releases consume the published crates.io facade at an
  exact pin. Hayashi's own Cargo features gate estimator dispatch modules in
  `src/lang/interpreter`.

Public surface:
- `src/lib.rs` exposes `Interpreter` and `run_source`. It also provides
  WASM bindings (`run_hayashi`, `set_print_callback`).
- `src/main.rs` is the `hay` binary: REPL, script runner, `dap`,
  `validate`, `install`, `dist-update`, `help`.
- `src/bin/hay-run.rs` is a headless, network-enabled runner.
- `src/bin/hay-kernel/` is the Jupyter kernel.

## 2. Core language

### 2.1 Lexer / Parser

- `src/lang/lexer.rs` tokenises source into `Token`. Notable tokens:
  `TsLag(usize)`, `TsLead(usize)`, `TsDiff(usize)` for `L2.x`, `F.y`,
  `D.z`; `FStringLit` for `f"..."`; contextual keywords are not reserved
  (`match` can be a variable name).
- `src/lang/parser.rs` is a hand-written recursive-descent / Pratt parser.
  Precedence (low to high): `||`, `&&`, comparisons, `+`/`-`, `*`/`/`/`%`,
  `^`/`**` (right-associative), unary, primary.
- `src/lang/ast.rs` defines `Expr`, `Stmt`, `Formula`, `RhsTerm`, `Spanned`.
  Formulas carry `lhs`, `rhs: Vec<RhsTerm>`, and `fe: Vec<String>` for
  `| id + time` fixed-effects syntax.

### 2.2 Interpreter

- `src/lang/interpreter.rs` holds `Interpreter` state:
  `env: Env`, `ts_info`, `panel_info`, `preserved`, `stored_models`,
  `rng`, `plugins`, `capturing`, `auto_display`, plus `DebugState`.
- `src/lang/interpreter/env.rs` implements block-scoped `Env` with a stack
  of `Scope`s. `let` is mutable; `const` and function parameters are not.
  Shadowing is forbidden.
- `src/lang/interpreter/value.rs` defines `Value` (primitives, `DataFrame`,
  `List`, `Dict`, `Series`, `UserFn`, 50+ model result variants, and a
  generic `ModelResult`). Shared values and model results use `Arc`, and
  `Model` requires `Send + Sync`, making `Value` structurally thread-safe.
  `Value::is_send_safe()` and `SendValue` preserve the **capture policy**;
  `ParallelItem` accepts all **iterator items**, including shared models.
- `src/lang/interpreter/eval_expr.rs` evaluates expressions, including
  f-strings, closures, `match`, if-expressions, pipes, ranges, and
  vectorised `eval_col_expr` / `eval_col_expr_typed` over DataFrames.
- `src/lang/interpreter/execution.rs` executes statements: `let`, `const`,
  `generate`, `replace`, `load`, `print`, `tsset`, `if`, `for`, `while`,
  `try/catch/finally`, `parallel for` via `std::thread::scope`.

### 2.3 Dispatch

`src/lang/interpreter/dispatch.rs` is the function-call router:
1. Plugin namespaces (`plugin::` from `.so`/`.wasm` or `hay install`).
2. Category-specific `eval_call_*` methods:
   `visualization`, `estimators_misc`, `estimators_timeseries`,
   `data_manipulation`, `post_estimation_ts`, `descriptive_lang`,
   `estimators_panel`, `estimators_micro`, `builtins`.
3. Scalar `greeners::Transforms::apply` / `apply2` for math functions.
4. User-defined functions (`fn` / closures).

## 3. Greeners integration

- `Cargo.toml` pins `greeners = { version = "=2.0.2-dev", default-features = false }`
  and, on `dev`, a `[patch.crates-io]` block resolves `greeners` and all
  `greeners-*` sub-crates from the Greeners repository `dev` branch. The local
  `crates/greeners` facade is no longer wired into the build;
  `default-features = false` only drops the facade's `url` feature. The
  `greeners-*` features are empty lists that gate Hayashi's own dispatch
  modules. Stable release branches remove the patch and consume the published
  crates.io facade at an exact pin.
- The `=` pin plus a committed `Cargo.lock` and `--locked` in CI mean a Greeners
  fix (including one that changes numbers) can only reach Hayashi through a
  deliberate commit: either `cargo update -p greeners` (moving the git `dev`
  head forward) or a facade bump in a dedicated PR that runs the full
  validation matrix. Pre-releases (`-rc`, `-dev`) are intentional on `dev`
  only — stable releases never depend on them.
- `src/lang/interpreter.rs` has `materialize_formula()` (lines ~1292–1441).
  It turns a Hayashi `Formula` into a Greeners `GFormula` by:
  1. Evaluating complex RHS terms (`log(K)`, `I(x^2)`) and inserting them
     as temporary columns (`__term_N`).
  2. Materialising interactions (`:`) into `__inter_N` columns.
  3. Delegating `C(var)` dummy expansion to Greeners.
  4. Returning `(Arc<DataFrame>, GFormula, display_names)`.
- `src/lang/interpreter/models.rs` contains dedicated wrappers that keep
  extra metadata for post-estimation / DAP:
  `OlsModel`, `BinaryModel`, `SurModel`, `PcaModel`, `FactorModel`,
  `DFMModel`, `ThreeSLSModel`, `PenalizedModel`.
- Most other estimators wrap `Arc<greeners::*Result>` directly.
- `src/lang/interpreter/dap/model_expansion/` converts model results into
  DAP / JSON trees (`coefficients` DataFrame, `fit` Dict, per-parameter
  Series).

### 3.1 Coupling risks

- All estimators call `df.to_design_matrix(&g_formula)`; any Greeners
  API change breaks many commands.
- `resolve_cov_full()` (`helpers/covariance.rs`) is duplicated across
  estimators.
- `display_names` (for UI) and `variable_names` (for Greeners) are two
  parallel naming systems that can drift.
- Dedicated model results remain excluded from **environment captures**
  by policy, but may be passed as **iterator items**, including repeated
  references and models nested in containers. `Arc` and `Model: Send + Sync`
  make this safe without assuming unique ownership. The capture check also
  traverses `Series.values` and every `ModelResult` value map (`fit`,
  `extras`, `fields`). See `ParallelItem` in `value.rs` and `control.md`.

## 4. I/O and DataFrames

- `src/io/` has loaders/export writers for CSV/TSV, DTA, Excel, Parquet,
  SQLite, ODBC, and URL fetching.
- All loaders accept `columns=` projection and `where=` row predicates.
- `src/lang/predicate.rs` defines `RowPredicate` (col-literal comparisons,
  `in [...]`, `!`, `&&`, `||`).
- **Parquet**: real pushdown — row-group pruning via min/max statistics,
  Arrow predicate pushdown, and `ProjectionMask`.
- **SQLite**: pushdown by rewriting the query to `SELECT ... WHERE ...`.
- **CSV / DTA / Excel**: row-by-row filter and column projection during
  accumulation; full file is still parsed.
- DataFrames are wrapped in `Arc<greeners::DataFrame>` and mutated with
  `Arc::make_mut` (copy-on-write).

## 5. Tests and validation

- `tests/smoke.rs` (14 229 lines, 776 tests) covers margins, parser,
  scoping, expressions, data manipulation, estimation, post-estimation,
  and finance examples.
- `tests/dap.rs` has 5 tests, all active and passing (no `#[ignore]`).
- `tests/numerical_golden.rs` checks OLS coefficients and SEs against
  statsmodels with `1e-8` tolerance.
- `tests/error_messages_english.rs` guards the English error-style policy.
- `validation/` runs Hayashi scripts against R/Python/Stata reference
  implementations. `validation/run.py` orchestrates discovery, execution,
  parsing, and tolerance-based comparison; `validation/matrix.yml` is the
  registry; `validation/MATRIX.md` is the human-readable dashboard.
- Validation metadata: `python validation/run.py --check` currently
  discovers 234 cases and `validation/matrix.yml` reports 219 `pass`,
  15 `not-supported`, 0 `blocked`, 0 `fail`.

### 5.1 CI

- `ci.yml`: validation metadata check, `cargo deny check advisories`,
  `cargo fmt --check`, `cargo clippy -- -D warnings`,
  `scripts/check_readme_counts.sh` (README test/validation/example counts),
  `cargo test` on Ubuntu/macOS/Windows. On `dev`, Greeners is resolved from the
  git `dev` branch via `[patch.crates-io]` at the commit frozen in
  `Cargo.lock`; CI runs with `--locked`, so any lockfile drift (including a
  moved Greeners head) fails the build instead of being silently resolved.
- `validation.yml`: full empirical validation on Ubuntu/macOS/Windows with
  R + Python.
- `nightly.yml`: builds `dev` and publishes a `nightly` GitHub release.
- `release.yml`: builds and publishes on `v*` tags.
- `docs.yml`: builds and deploys mdBook to GitHub Pages on `master`.

## 6. Branches and releases

- `dev`: active development. Tracks Greeners `dev` branch (pre-release
  `2.0.2-dev`) via `[patch.crates-io]`.
- `master`: stable releases. Consumes the published `greeners` facade from
  crates.io at an exact pin (e.g. `=2.0.1` in 0.2.10-1). Tags include
  `v0.2.0` through `v0.2.10-1` and `nightly`.
- `gh-pages`: published site.
- Open feature/fix branches:
  - `pr62` — validation comparison against every reference.
  - `pr-charles-52` / `pr-charles-50` — golden-label normalisation and
    Fama-MacBeth simulated case.
  - `docs/fix-estimator-command-drift` — docs alignment.
  - `feature/margins-binary-vcov` — binary margins MLE covariance.
  - `bugfix/rnormal-normal-draws` — `rnormal` standard-normal fix.
  - `bugfix/sqlite-identifier-quoting` — SQLite quoting hardening.
  - `tests/golden-ols-se` — OLS golden tests.


Workflow: PRs target `dev`; `master` is release-only.

## 7. Known fragile areas

1. **Model wrapper inconsistency**: some estimators use dedicated
   wrappers, others use `Arc<GreenersResult>` or `ModelResult`. Uniform
   access to residuals / predict requires knowing which variant is which.
2. **DAP tests active**: all 5 DAP tests run and pass with
   `cargo test --test dap` (no `#[ignore]`).
3. **`fe()` `cluster=` support**: `cluster=` and `cluster2=` are supported
   via `resolve_cov_full()`. Validation case `panel_fe_cluster_wagepan`
   confirms the feature.
4. **Validation only on Ubuntu**: R/Python reference numbers can differ
   slightly across platforms.
5. **Error-language migration**: user-facing strings are being migrated
   from Portuguese to English (`docs/error-style.md` governs style; CI
   guard `scripts/check_pt_errors.sh` checks newly added lines).

## 8. Design decisions, not fragilities

- **Greeners dependency**: on `dev`, Hayashi resolves `greeners` (pinned
  `=2.0.2-dev`) from the Greeners `dev` branch through `[patch.crates-io]`;
  stable releases consume the published crates.io facade at an exact pin.
  Either way the resolution is frozen in `Cargo.lock` with `--locked` in CI:
  Greeners moves on its own release train and Hayashi follows it only when
  someone deliberately bumps the pin or the git head.
- **`parallel for` and shared model results**: all `Value` variants use
  owned data or `Arc`, and `Model` requires `Send + Sync`. `SendValue`,
  `ParallelItem` and `ThreadResult` derive thread safety from their fields;
  none uses `unsafe impl Send/Sync`. Model aliases may occur across items,
  nested containers, returned results and the parent, so moving a wrapper
  must never be treated as proof of exclusive ownership. Dedicated models
  remain iterator-only by capture policy. Regression checks:
  `cargo test --lib parallel_` and `cargo test --test smoke parallel_for_`.

## 9. Adding a new estimator

Typical steps:
1. Implement the algorithm in `Greeners` (on `develop`).
2. In Hayashi (`dev`):
   - Add a `Value` variant in `src/lang/interpreter/value.rs` if needed.
   - Add a wrapper in `src/lang/interpreter/models.rs` if extra metadata
     (residuals, variable names) must be preserved.
   - Add dispatch in `src/lang/interpreter/estimators_*.rs`.
   - Add DAP expansion in `src/lang/dap/model_expansion/`.
   - Add `tidy()`/`glance()` support in `builtins/tidy.rs` / `glance.rs`.
   - Add smoke tests in `tests/smoke.rs` and a validation case in
     `validation/cases/<id>/` if a reference implementation exists.
   - Update `help()` topics and examples if user-facing.

## 10. Git identity

When commits are made in this environment, the canonical author/committer email is:
- `flavio.vcorrea@ufpel.edu.br`

## 11. Validation reference policy

Every validation case must declare which reference implementations it runs
against, and the reason must be explicit when both R and Python are not
available:

- `pass` is recorded when at least one declared reference (R, Python, Stata,
  or an independent implementation) runs and Hayashi matches it within the
  declared tolerances.
- `not-supported` is recorded when no deterministic, reproducible reference
  can be made to work in CI. The reason is written in the case `notes` and
  reflected in `matrix.yml` and `MATRIX.md`.
- `partial` is allowed on macOS and Windows because those runners have more
  fragile R and Python environments, but `partial` is not accepted on Ubuntu
  unless `--allow-partial` is explicitly passed.

A case is never left silently unvalidated. If both R and Python references
exist but one of them fails, that is recorded as `partial` (or `not-supported`
if the failure is not fixable). If a reference does not exist at all, the
case is marked `not-supported` and the missing reference is documented.

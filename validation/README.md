# Hayashi Empirical Validation Programme

This directory contains a systematic, reproducible validation programme for
Hayashi against trusted reference implementations (R, Python/statsmodels,
optionally Stata).  The goal is to provide durable, versioned evidence that
Hayashi produces correct results under realistic applied-econometrics
workflows.

## Core principle

Every validation case is versioned, auditable, and fully automated from the
command line.  The evidence lives in the repository as manifests, scripts,
expected outputs, and comparison reports.

## Quick start

```bash
python validation/run.py
```

The orchestrator reads `validation/matrix.yml`, runs the Hayashi script and
the reference scripts for each selected case, compares the declared quantities
against tolerances, and updates `MATRIX.md`.

For focused development or issue triage, run a single case without updating
the generated matrix files:

```bash
python validation/run.py --case heckman_mroz --no-write
```

Useful runner options:

```bash
python validation/run.py --list
python validation/run.py --check
python validation/run.py --case heckman_mroz
python validation/run.py --case heckman_mroz --case ols_cluster_wagepan --no-write
```

Use `--check` before adding or editing cases. It validates the case registry,
case metadata, declared script paths, and generated `MATRIX.md` without running
Hayashi or reference estimators.

Exit codes:

- `0` — selected cases completed with no failed, blocked, partial, or not-started result.
- `0` — metadata is consistent when `--check` is passed.
- `0` — validation blocked, but `--allow-blocked` was passed.
- `1` — at least one case failed, was blocked without `--allow-blocked`, was partial without `--allow-partial`, or was not started.
- `1` — metadata is inconsistent when `--check` is passed.

The same options can be passed through `hay validate`, for example
`hay validate --case heckman_mroz --no-write`.

## Requirements

- Hayashi CLI (`hay`) built from the repository.
- R with `Rscript` and every package listed in `DESCRIPTION`.
- Python 3 with the packages listed in `requirements.txt`.
- Stata is optional and only used when `stata` is found in `$PATH`.

The workflow pins the Greeners numerical-engine revision. The R and Python
package sets are declared here but are not yet version-locked; #140 tracks the
lockfile policy and CI restoration needed for a fully pinned reference
environment.

Install Python dependencies:

```bash
pip install -r validation/requirements.txt
```

Install R dependencies:

```bash
Rscript -e 'd <- read.dcf("validation/DESCRIPTION"); pkgs <- trimws(strsplit(d[1, "Imports"], ",")[[1]]); install.packages(pkgs, repos="https://cloud.r-project.org/")'
```

## Reference implementation coverage

Most cases provide both an R and a Python reference implementation. The
validation runner runs every declared reference and compares Hayashi
independently against every reference that runs successfully. A case is
`blocked` when no declared reference runs. It is `partial` when at least one
reference runs but another declared reference fails or is missing; partial
results exit non-zero unless `--allow-partial` is passed. `MATRIX.md` lists the
declared references and recorded case status. When a runner result records
per-reference execution details, `MATRIX.md` renders them in the Reference
column. A few cases implement the
estimator manually in base R/Python because no suitable packaged reference is
available; each case's `README.md` documents the exact packages and
implementation choices.

## Cross-platform CI policy

The validation workflow runs on Ubuntu, macOS and Windows.

- **Ubuntu is the precision reference.** Both R and Python references run
  reliably, and a `pass` on Ubuntu means Hayashi agrees with both references
  within the declared tolerances. This is the strongest evidence of numerical
  correctness.

- **macOS and Windows are treated as compatibility smoke tests.** The runner
  uses `--allow-partial` on these platforms because the reference environments
  are more fragile:
  - **macOS**: the R `renv` snapshot sometimes fails to restore or compile
    native packages (e.g. `plm`, `wooldridge` dependencies), causing the R
    reference to halt without output. When the Python reference runs and
    Hayashi matches it, the case is recorded as `partial`.
  - **Windows**: the reference scripts and the Hayashi scripts use Unix-only
    idioms (`/dev/stdout`, repository-relative forward-slash paths, and UTF-8
    console output such as `Δvalue`). The runner rewrites `run.hay` on the fly
    to use concrete output files and absolute paths, and it forces UTF-8 I/O
    decoding. Any remaining `partial` results on Windows are due to reference
    scripts that fail for environment reasons (e.g. `UnicodeEncodeError` on
    `cp1252` consoles) rather than numerical disagreement.

A `partial` result on macOS or Windows still means Hayashi matched every
available reference on that platform; it does not indicate a precision problem
in Hayashi.

## Directory layout

```text
validation/
  README.md              # this file
  matrix.yml             # machine-readable case registry
  MATRIX.md              # human-readable dashboard
  run.py                 # CLI orchestrator
  requirements.txt       # Python dependencies
  DESCRIPTION            # R dependencies (for renv/pak)
  templates/             # templates for new cases
  cases/                 # one directory per validation case
    <case-id>/
      case.yml           # case metadata, status, tolerances
      README.md          # human-readable description
      data/              # small datasets or download instructions
      hayashi/
        run.hay          # Hayashi script
        output.json      # produced by run.hay
      reference/
        run.R            # R reference script
        run.py           # Python reference script
        run.do           # optional Stata reference script
        expected.json    # produced by reference scripts
```

## Status values

`case.yml` declares whether the runner should execute a case. In the current
schema, `pass` means the case is active and runnable; `blocked`,
`not-supported`, and `not-started` cause the runner to skip it. A selected
`not-started` case exits non-zero, so an unimplemented case cannot create a
green validation result. `matrix.yml`
records the last observed status and is regenerated after a run. The runner
retains the manifest status before merging the recorded matrix status so a stale
result cannot alter execution eligibility. The metadata check rejects
contradictions such as a `not-started` case with a recorded `pass` result.

- `pass` — the recorded run matched reference within declared tolerances.
- `fail` — Hayashi differs from reference beyond tolerances; an issue should
  be opened.
- `blocked` — cannot be evaluated because a required dependency, feature, or
  known defect prevents execution; link a repository defect to its issue.
- `not-supported` — the validation programme cannot currently test the stated
  estimator/workflow contract. This does not necessarily mean Hayashi lacks
  the command.
- `not-started` — case is registered but not implemented.

## Adding a new case

1. Copy `validation/templates/case.yml` and `validation/templates/README.md`.
2. Fill in dataset source, estimator family, reference software, quantities,
   tolerances, and status.
3. Write `hayashi/run.hay`, `reference/run.R`, and `reference/run.py`.
   For book-based cases, generate the dataset from the same DGP used in
   `book_pt_BR/codes/*.hay` (or `book_en/codes/*.hay`) so the reference
   implementation can reproduce the exact series.
4. Ensure each script emits the comparable output on `stdout` (and optionally
   writes it to `cases/<id>/reference/expected.*` and
   `cases/<id>/hayashi/output.*` for debugging).
5. Optional: add an entry to `matrix.yml` with `id`, `dimension`, and `notes`.
   If omitted, `hay validate` auto-discovers the case from the filesystem.
6. Run `python validation/run.py --check`, then `hay validate`, and commit the
   updated `MATRIX.md`.

### Book-based simulated cases

Cases derived from the Hayashi book use the same DGP as the corresponding
chapter script. The reference implementation should replicate the estimator
used by Hayashi (e.g., Hannan-Rissanen for the default `arima()` path) so that
coefficients match exactly. Coefficients and standard errors are compared
when the reference can reproduce the same inference; otherwise only
coefficients are compared with a documented rationale.

## Methodological guardrails

- Datasets must be public and have a clear licence or redistribution status.
- Each case is recorded in `matrix.yml` before selection; cases are never
  silently dropped.
- Tolerances must be declared with a rationale.
- Failures and blocked cases are recorded as first-class outcomes.

## References

This programme follows reproducible-research practice drawn from:

- The Turing Way reproducible-research guide
- AEA data and code policy
- ACM artifact review and badging policy

## Scope

The validation programme covers four dimensions:

1. **Numerical correctness** — coefficients, standard errors, likelihoods,
   marginal effects, fit statistics.
2. **Real-data robustness** — missing values, strings/categoricals, dates,
   unbalanced panels, collinearity, singleton groups, non-finite values,
   convergence failures, awkward variable names.
3. **Applied workflow fit** — import, cleaning, estimation, diagnostics,
   prediction, export.
4. **Runtime reliability** — no panics, hangs, silent miscomputations, or
   misleading results.

## Estimators not covered by validation

Not every command in Hayashi is a validation case. The programme focuses on
core empirical estimators and intentionally excludes some command categories:

- **Diagnostic/test commands without a dedicated case** (e.g., `adf`, `kpss`,
  `engle_granger`, `johansen`, `bgodfrey`, `archtest`, `hausman`) — validated
  indirectly through the estimators that use them. Diagnostics with a case,
  such as `granger`, `ljungbox`, `white`, and `reset`, are listed in the matrix.
- **Utility/data manipulation commands** (e.g., `generate`, `filter`,
  `summarize`, `load`, `export`) — covered by `cargo test`, not by empirical
  validation.
- **Visualization commands** (e.g., `plot`, `scatter`, `histogram`) — not part
  of numerical validation.
- **Niche or hard-to-reference estimators** (e.g., `portfolio_sort`,
  `double_sort`, dynamic factor, GAM, multiple imputation) —
  require specialised datasets or lack canonical open-source reference
  implementations.
- **Estimators with output format limitations** (e.g., `svar`, `svec`) —
  require changes to the Greeners export format before they can be parsed.

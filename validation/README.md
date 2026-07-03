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

## Requirements

- Hayashi CLI (`hay`) built from the repository.
- R with `Rscript` and the `wooldridge` package.
- Python 3 with the packages listed in `requirements.txt`.
- Stata is optional and only used when `stata` is found in `$PATH`.

Install Python dependencies:

```bash
pip install -r validation/requirements.txt
```

Install R dependencies:

```bash
Rscript -e 'install.packages("wooldridge", repos="https://cloud.r-project.org/")'
```

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

- `pass` — Hayashi matches reference within declared tolerances.
- `fail` — Hayashi differs from reference beyond tolerances; an issue should
  be opened.
- `blocked` — cannot run because of a missing feature or bug; link to issue.
- `not-supported` — the estimator/workflow is not supported by Hayashi yet.
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
6. Run `hay validate` and commit the updated `MATRIX.md`.

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

- **Diagnostic/test commands** (e.g., `adf`, `kpss`, `granger`,
  `engle_granger`, `johansen`, `ljungbox`, `white`, `reset`, `bgodfrey`,
  `archtest`, `hausman`) — validated indirectly through the estimators that use
  them.
- **Utility/data manipulation commands** (e.g., `generate`, `filter`,
  `summarize`, `load`, `export`) — covered by `cargo test`, not by empirical
  validation.
- **Visualization commands** (e.g., `plot`, `scatter`, `histogram`) — not part
  of numerical validation.
- **Niche or hard-to-reference estimators** (e.g., `portfolio_sort`,
  `double_sort`, Fama-MacBeth, dynamic factor, GAM, multiple imputation) —
  require specialised datasets or lack canonical open-source reference
  implementations.
- **Estimators with output format limitations** (e.g., `svar`, `svec`) —
  require changes to the Greeners export format before they can be parsed.

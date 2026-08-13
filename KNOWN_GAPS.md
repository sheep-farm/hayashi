# Known validation and documentation gaps for the 0.2.10 release

This document tracks estimator families that are implemented and exposed to users but are not yet covered by the empirical validation programme in `validation/`, or where the existing coverage is thin enough that release notes should be careful.

## Status of the validation programme

- As of this update, `validation/matrix.yml` contains 226 cases.
- 200 cases are currently `pass`; 14 are documented as `not-supported` and 1 as `blocked`.
- The cases compare Hayashi output against R and/or Python reference implementations (statsmodels, linearmodels, wooldridge, etc.) with family-specific tolerances.

## Implemented estimators not yet in the validation matrix

These commands exist in the interpreter dispatch and are listed in user-facing docs, but have no corresponding case under `validation/cases/`:

- `svec` — currently shares the same dispatch path as `svar` (`"svar" | "svec"`), so it is treated as a Cholesky-SVAR alias rather than a distinct validation target.

The following estimators were previously in this list but are now covered by validation cases (`feiv_simulated`, `clogit_simulated`, `cpoisson_simulated`, `varma_simulated`, `threesl_simulated`):

- `feiv` — fixed-effects IV (`src/lang/interpreter/estimators_panel.rs`)
- `clogit` — conditional logit (`src/lang/interpreter/estimators_panel.rs`)
- `cpoisson` — conditional Poisson / PPML (`src/lang/interpreter/estimators_panel.rs`)
- `varma` / `varmax` — vector ARMA (`src/lang/interpreter/estimators_timeseries.rs`)
- `three_sls` / `threesl` / `3sls` / `reg3` — three-stage least squares (`src/lang/interpreter/estimators_timeseries.rs`)

## Validated but with thin or convention-sensitive coverage

These families appear in the matrix, but the coverage should be described carefully in release notes:

- `kalman` — one R-only local-level case (`kalman_nyse`).
- `svar` — one Cholesky-identified SVAR case. Broader SVAR/SVEC identification and export/parse support may still need work.
- `ab` (Arellano-Bond), `sysgmm` (Blundell-Bond), `pcse`, `xtgls` — dynamic-panel/GMM and panel-GLS cases are present, but the references implement the same two-step procedures used by Hayashi/Greeners; other packages (e.g. `plm::pgmm`) use different instrument and weighting conventions, so results may not be portable across implementations.
- `km` (Kaplan-Meier), `cox`, `psm`, `synth`, `did`, `rd`, `fuzzy_rd`, `qreg`, `rlm`, `gee`, `gmm`, `iv` — have at least one case, but often rely on simulated or selected real datasets; broader coverage is desirable.

## Implemented but not yet validated

- `be` (between estimator) is implemented in `src/lang/interpreter/estimators_panel.rs` and exposed through `help()`, but has no empirical validation case yet. It should be treated as an unvalidated command until a reference case is added.

## Recommendation for release notes

Do not claim that "every estimator listed in the documentation has been validated against R and Python". Use wording like:

- "0.2.10 ships with 123 validated empirical cases covering OLS, IV, panel, time-series, binary, count, regularization, causal, and ML families."
- "Several families are implemented and documented but await dedicated validation cases; see `KNOWN_GAPS.md`."
- "Some panel and time-series estimators are convention-sensitive; the validation cases explicitly document the reference implementation and conventions used."

## Follow-up work

- Confirm that `svec` is intended solely as a Cholesky-SVAR alias, or implement/validate it as a distinct estimator if required.
- Implement and validate `be` before restoring it to the public command list.
- Expand thin cases (`kalman`, `svar` with non-Cholesky identification, broader dynamic panel references) where practical.

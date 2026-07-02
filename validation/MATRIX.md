# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| OLS | Wooldridge `wage1` | R / Python | not-started | — | wage ~ educ + exper + tenure |

## Status legend

- `pass` — Hayashi matches reference within declared tolerances.
- `fail` — Hayashi differs from reference beyond tolerances.
- `blocked` — cannot run because of a missing feature or bug.
- `not-supported` — estimator/workflow not supported yet.
- `not-started` — registered but not implemented.

This matrix is generated from `validation/matrix.yml` by `validation/run.py`.

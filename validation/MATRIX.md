# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| iv | wooldridge::card | R, Python | pass | — | IV with education endogenous and nearc4 as instrument. |
| logit | wooldridge::mroz | R, Python | pass | — | Logit labour-force participation on the Mroz dataset. |
| ols | wooldridge::wage1 | R, Python | pass | — | First real-dataset validation case. |
| panel_fe | wooldridge::grunfeld | R, Python | pass | — | Panel fixed-effects investment demand model (Grunfeld). |
| poisson | wooldridge::fertil2 | R, Python | pass | — | Poisson regression for number of children on the fertil2 dataset. |
| probit | wooldridge::mroz | R, Python | pass | — | Probit labour-force participation on the Mroz dataset. |
| wls | wooldridge::hprice1 | R, Python | pass | — | WLS with weights generated inside Hayashi to avoid sandbox file issues. |

## Status legend

- `pass` — Hayashi matches reference within declared tolerances.
- `fail` — Hayashi differs from reference beyond tolerances.
- `blocked` — cannot run because of a missing feature or bug.
- `not-supported` — estimator/workflow not supported yet.
- `not-started` — registered but not implemented.

This matrix is generated from `validation/matrix.yml` by `validation/run.py`.


# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| ab | wooldridge::grunfeld | R | pass | — | Arellano-Bond difference GMM for dynamic panel investment demand. |
| ardl | statsmodels::macrodata | R, Python | pass | — | ARDL(1,1) model of US real GDP on consumption. |
| arima | statsmodels::macrodata | R, Python | pass | — | ARIMA(1,1,1) on log US real GDP via exact Gaussian MLE. |
| autoreg | statsmodels::macrodata | R, Python | pass | — | AR(1) on US real GDP with constant and trend. |
| cox | statsmodels::heart | R, Python | pass | — | Cox proportional hazards regression for survival time after heart transplant. |
| did | wooldridge::kielmc | R, Python | pass | — | Difference-in-differences effect of incinerator proximity on log house prices. |
| ets | statsmodels::macrodata | R, Python | pass | — | Exponential smoothing state-space model on US real GDP. Blocked because Hayashi uses SSE grid search while references use MLE. |
| garch | wooldridge::nyse | R, Python | pass | — | GARCH(1,1) on NYSE returns. |
| glsar | wooldridge::hprice1 | R, Python | pass | — | GLS with AR(1) errors on housing price equation. |
| gmm | wooldridge::card | R, Python | pass | — | GMM returns to schooling with nearc4 as instrument for education. |
| iv | wooldridge::card | R, Python | pass | — | IV with education endogenous and nearc4 as instrument. |
| lasso | wooldridge::hprice1 | R, Python | pass | — | Lasso regression of house price on lot size, square footage and bedrooms. |
| logit | wooldridge::mroz | R, Python | pass | — | Logit labour-force participation on the Mroz dataset. |
| ols | wooldridge::wage1 | R, Python | pass | — | First real-dataset validation case. |
| panel_fe | wooldridge::grunfeld | R, Python | pass | — | Panel fixed-effects investment demand model (Grunfeld). |
| poisson | wooldridge::fertil2 | R, Python | pass | — | Poisson regression for number of children on the fertil2 dataset. |
| probit | wooldridge::mroz | R, Python | pass | — | Probit labour-force participation on the Mroz dataset. |
| qreg | wooldridge::wage1 | R, Python | pass | — | Median quantile regression of wage on education, experience, and tenure. |
| re | grunfeld | R, Python | pass | — | Random-effects investment demand model (Grunfeld). |
| tobit | wooldridge::mroz | R, Python | pass | — | Tobit regression of hours worked with left censoring at zero. |
| var | statsmodels::macrodata | R, Python | pass | — | VAR(2) on US real GDP and consumption. |
| wls | wooldridge::hprice1 | R, Python | pass | — | WLS with weights generated inside Hayashi to avoid sandbox file issues. |

## Status legend

- `pass` — Hayashi matches reference within declared tolerances.
- `fail` — Hayashi differs from reference beyond tolerances.
- `blocked` — cannot run because of a missing feature or bug.
- `not-supported` — estimator/workflow not supported yet.
- `not-started` — registered but not implemented.

This matrix is generated from `validation/matrix.yml` by `validation/run.py`.


# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| ab | wooldridge::grunfeld | R | pass | — | Arellano-Bond difference GMM for dynamic panel investment demand. |
| arima | simulated_ar1 | R, Python | pass | — | Uses the same simulated AR(1) DGP as Chapter 26 of the book. |
| ardl | statsmodels::macrodata | R, Python | pass | — | ARDL(1,1) model of US real GDP on consumption. |
| arima | simulated_rw | R, Python | pass | — | ARIMA(1,1,0) on a simulated random walk with seed 42. Intercept is excluded from comparison because R/Python references are estimated without trend. |
| arima | statsmodels::macrodata | R, Python | pass | — | ARIMA(1,1,1) on log US real GDP via exact Gaussian MLE. |
| arima | simulated_arma11 | R, Python | pass | — | Uses the same simulated ARMA(1,1) DGP as Chapter 26 of the book. Intercept is excluded from comparison because Hayashi profiles it out in MLE (SE = 0). |
| autoreg | statsmodels::macrodata | R, Python | pass | — | AR(1) on US real GDP with constant and trend. |
| vecm | simulated_cointegrated | R, Python | pass | — | VECM(1) on a simulated cointegrated system where y = 2*x + e2 and x = cumsum(e1). Only the cointegration (beta) and adjustment (alpha) coefficients are compared. |
| cox | statsmodels::heart | R, Python | pass | — | Cox proportional hazards regression for survival time after heart transplant. |
| did | wooldridge::kielmc | R, Python | pass | — | Difference-in-differences effect of incinerator proximity on log house prices. |
| elasticnet | wooldridge::hprice1 | R, Python | pass | — | Elastic Net regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| ets | statsmodels::macrodata | R, Python | pass | — | Exponential smoothing state-space model on US real GDP. Blocked because Hayashi uses SSE grid search while references use MLE. |
| fmb | simulated_fmb_panel | R:passed, Python:passed * | pass | 49 | Classic Fama-MacBeth regression on a deterministic simulated asset panel. |
| garch | simulated_garch11 | Python | pass | — | Uses the same simulated GARCH(1,1) DGP as Chapter 30 of the book. MLE tolerances are looser because the optimizer may stop at slightly different points. |
| garch | wooldridge::nyse | R, Python | pass | — | GARCH(1,1) on NYSE returns. |
| glsar | wooldridge::hprice1 | R, Python | pass | — | GLS with AR(1) errors on housing price equation. |
| gmm | wooldridge::card | R, Python | pass | — | GMM returns to schooling with nearc4 as instrument for education. |
| heckman | wooldridge::mroz | R, Python | pass | — | Two-step Heckman (Heckit) on the Mroz dataset. SEs are approximate because the reference implementations are two-step. |
| iv | wooldridge::card | R, Python | pass | — | IV with education endogenous and nearc4 as instrument. |
| lasso | wooldridge::hprice1 | R, Python | pass | — | Lasso regression of house price on lot size, square footage and bedrooms. |
| logit | wooldridge::mroz | R, Python | blocked | 58 | Blocked real-dataset validation case for logit average marginal effects on the Mroz dataset; Hayashi AMEs match R/statsmodels, but delta-method SEs differ pending issue #58. |
| logit | wooldridge::mroz | R, Python | pass | — | Logit labour-force participation on the Mroz dataset. |
| arima | simulated_ma1 | R, Python | pass | — | Uses the same simulated MA(1) DGP as Chapter 26 of the book. |
| mlogit | AER::TravelMode | R, Python | pass | — | Multinomial logit of chosen travel mode (air=1, train=2, bus=3, car=4) on income, wait time, vehicle cost and travel time. Alternative-specific attributes are averaged per individual to make them individual-specific covariates. |
| negbin | wooldridge::fertil2 | R, Python | pass | — | Negative binomial regression for number of children on age, education, electric and urban indicators. Dispersion parameter (alpha) is not compared because Hayashi does not report it; coefficient tolerance is 2e-1 due to different alpha estimates. |
| logit | wooldridge::beauty | R, Python | pass | — | Ordered logit of looks (2, 3, 4) on female, educ, exper, black. |
| ols | wooldridge::wagepan | R, Python | pass | — | OLS wage equation with one-way cluster-robust standard errors by worker id. |
| ols | wooldridge::wage1 | R, Python | pass | — | First real-dataset validation case. |
| ols | wooldridge::wage1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 2 Example 2.10 log wage equation. |
| oprobit | wooldridge::beauty | R, Python | pass | — | Ordered probit model of self-reported beauty rating (looks 2-5) on female, education, experience and black indicators. |
| panel_fe | wooldridge::grunfeld | R, Python | pass | — | Panel fixed-effects investment demand model (Grunfeld). |
| poisson | wooldridge::fertil2 | R, Python | pass | — | Poisson regression for number of children on the fertil2 dataset. |
| probit | wooldridge::mroz | R, Python | pass | — | Probit labour-force participation on the Mroz dataset. |
| psm | wooldridge::jtrain3 | R, Python | pass | — | 1:1 nearest-neighbor propensity score matching with caliper 0.2 and bootstrap SE. |
| qreg | wooldridge::wage1 | R, Python | pass | — | Median quantile regression of wage on education, experience, and tenure. |
| rdd | rdd_book | R, Python | pass | — | Sharp RDD with local linear regression, triangular kernel and Imbens-Kalyanaraman bandwidth. |
| re | grunfeld | R, Python | pass | — | Random-effects investment demand model (Grunfeld). |
| ridge | wooldridge::hprice1 | R, Python | pass | — | Ridge regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| sur | wooldridge::grunfeld | R, Python | pass | — | Two-equation SUR (Zellner FGLS) on the Grunfeld investment data. |
| synth | synth_smoking | R, Python | pass | — | Synthetic-control ATT on a simulated panel with 10 donors and 1 treated unit. |
| tobit | wooldridge::mroz | R | pass | — | Tobit regression of hours worked with left censoring at zero. Hayashi matches AER::tobit at displayed precision; the custom Python MLE is retained as a diagnostic script but is not the active reference. |
| var | simulated_var1 | R, Python | pass | — | Uses the same simulated bivariate VAR(1) DGP as Chapter 28 of the book. |
| var | statsmodels::macrodata | R, Python | pass | — | VAR(2) on US real GDP and consumption. |
| wls | wooldridge::hprice1 | R, Python | pass | — | WLS with weights generated inside Hayashi to avoid sandbox file issues. |

## Status legend

- `pass` — Hayashi matches reference within declared tolerances.
- `fail` — Hayashi differs from reference beyond tolerances.
- `blocked` — cannot run because of a missing feature or bug.
- `not-supported` — estimator/workflow not supported yet.
- `not-started` — registered but not implemented.

The Reference column shows per-reference status as `name:status`,
where `*` marks the reference used for comparison. A declared
reference that fails or is missing blocks the case.

This matrix is generated from `validation/matrix.yml` by `validation/run.py`.

This matrix covers the core empirical estimators. Some commands are
intentionally excluded for the reasons described in the "Estimators not
covered by validation" section of the README.

Esta matriz abrange os estimadores empíricos centrais. Alguns comandos são
deixados de fora intencionalmente pelos motivos descritos na seção
"Estimators not covered by validation" do README.

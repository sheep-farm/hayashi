# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| ab | wooldridge::grunfeld | R:passed * | pass | — | Arellano-Bond difference GMM for dynamic panel investment demand. |
| arima | simulated_ar1 | R:passed *, Python:passed * | pass | — | Uses the same simulated AR(1) DGP as Chapter 26 of the book. |
| ardl | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | ARDL(1,1) model of US real GDP on consumption. |
| arima | simulated_rw | R:passed *, Python:passed * | pass | — | ARIMA(1,1,0) on a simulated random walk with seed 42. Intercept is excluded from comparison because R/Python references are estimated without trend. |
| arima | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | ARIMA(1,1,1) on log US real GDP via exact Gaussian MLE. |
| arima | simulated_arma11 | R:passed *, Python:passed * | pass | — | Uses the same simulated ARMA(1,1) DGP as Chapter 26 of the book. Intercept is excluded from comparison because Hayashi profiles it out in MLE (SE = 0). |
| autoreg | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | AR(1) on US real GDP with constant and trend. |
| betareg | wooldridge::401k | R:passed * | pass | — | Beta regression on 401k participation rates. Greeners estimates by BFGS with an analytic gradient and matches R betareg. |
| causal_impact | simulated_causal_impact | R:passed *, Python:passed * | pass | — | Bayesian structural time series for counterfactual inference (Brodersen 2015). Uses simulated data with known treatment effect. |
| cloglog | wooldridge::affairs | R:passed *, Python:passed * | pass | — | Complementary log-log GLM on Wooldridge affairs. Fixed cloglog link derivative sign; Hayashi now converges and matches R glm. |
| vecm | simulated_cointegrated | R:passed *, Python:passed * | pass | — | VECM(1) on a simulated cointegrated system where y = 2*x + e2 and x = cumsum(e1). Only the cointegration (beta) and adjustment (alpha) coefficients are compared. |
| cox | statsmodels::heart | R:passed *, Python:passed * | pass | — | Cox proportional hazards regression for survival time after heart transplant. |
| dcc_garch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | DCC-GARCH (Dynamic Conditional Correlation GARCH) on NYSE returns. Uses simplified DCC-GARCH(1,1) model. |
| did | wooldridge::kielmc | R:passed *, Python:passed * | pass | — | Difference-in-differences effect of incinerator proximity on log house prices. |
| double_ml | simulated_double_ml | R:passed *, Python:passed * | pass | — | Double Machine Learning (Chernozhukov et al. 2018) for heterogeneous treatment effects. Uses simulated data with known treatment effect. |
| egarch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | EGARCH(1,1) on NYSE returns. |
| elasticnet | wooldridge::hprice1 | Python:passed * | pass | — | Elastic Net regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| ets | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | Exponential smoothing state-space model on US real GDP. Blocked because Hayashi uses SSE grid search while references use MLE. |
| fmb | simulated_fmb_panel | R:passed *, Python:passed * | pass | 49 | Classic Fama-MacBeth regression on a deterministic simulated asset panel. |
| ftest_robust | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Robust F-test (Wooldridge 2010) with cluster-robust covariance for joint significance test. |
| garch | simulated_garch11 | Python:passed * | pass | — | Uses the same simulated GARCH(1,1) DGP as Chapter 30 of the book. MLE tolerances are looser because the optimizer may stop at slightly different points. |
| garch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | GARCH(1,1) on NYSE returns. |
| gee | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 generalized estimating equations wage equation. |
| glm | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Poisson GLM for number of children on Wooldridge fertil2. |
| glsar | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | GLS with AR(1) errors on housing price equation. |
| gmm | wooldridge::card | R:passed *, Python:passed * | pass | — | GMM returns to schooling with nearc4 as instrument for education. |
| hausman_robust | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Robust Hausman test (Cameron-Trivedi 2005, Wooldridge 2010) with cluster-robust covariance. |
| heckman | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Two-step Heckman (Heckit) on the Mroz dataset. SEs are approximate because the reference implementations are two-step. |
| iv | wooldridge::card | R:passed *, Python:passed * | pass | — | IV with education endogenous and nearc4 as instrument. |
| iv | wooldridge::card | R:passed *, Python:passed * | pass | 97 | IV returns-to-schooling equation with one-way clustered standard errors by Census region. |
| iv | wooldridge::mroz | R:passed *, Python:passed * | pass | 95 | IV returns-to-schooling equation with HC1 heteroskedasticity-robust standard errors. |
| iv | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 15 Example 15.1 IV returns to schooling for married women. |
| kalman | wooldridge::nyse | R:passed * | pass | — | Local-level Kalman filter on NYSE returns. Hayashi estimates sigma_obs and sigma_state by MLE and returns a printable result object. |
| kmeans | simulated_kmeans | R:passed *, Python:passed * | pass | — | K-Means clustering (MacQueen 1967) with k-means++ initialization. Uses simulated 2D data with 3 Gaussian clusters. |
| lasso | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | Lasso regression of house price on lot size, square footage and bedrooms. |
| logit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Logit average marginal effects on Wooldridge mroz. |
| logit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Logit labour-force participation on the Mroz dataset. |
| did | simulated_absorbing_panel | Python:passed * | pass | — | LP-DiD quickstart against pylpdid on an absorbing staggered-adoption panel. R reference is left aside for now. |
| arima | simulated_ma1 | R:passed *, Python:passed * | pass | — | Uses the same simulated MA(1) DGP as Chapter 26 of the book. |
| mice_chained | simulated_mice | R:passed *, Python:passed * | pass | — | MICE (Multiple Imputation by Chained Equations, van Buuren 2011) with m=5, iter=10. Uses simulated data with MCAR missing values. |
| mixed | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 mixed linear model wage equation. |
| mlogit | AER::TravelMode | R:passed *, Python:passed * | pass | — | Multinomial logit of chosen travel mode (air=1, train=2, bus=3, car=4) on income, wait time, vehicle cost and travel time. Alternative-specific attributes are averaged per individual to make them individual-specific covariates. |
| negbin | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Negative binomial regression for number of children on age, education, electric and urban indicators. Dispersion parameter (alpha) is not compared because Hayashi does not report it; coefficient tolerance is 2e-1 due to different alpha estimates. |
| ologit | wooldridge::beauty | R:passed *, Python:passed * | pass | — | Ordered logit of looks (2, 3, 4) on female, educ, exper, black. |
| ols | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | OLS wage equation with one-way cluster-robust standard errors by worker id. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | 89 | OLS log-wage equation with HC3 heteroskedasticity-robust standard errors. |
| ols | wooldridge::phillips | R:passed *, Python:passed * | pass | 91 | OLS expectations-augmented Phillips curve with Newey-West HAC standard errors. |
| ols | wooldridge::wagepan | R:passed *, Python:passed * | pass | 87 | OLS wage equation with two-way clustered standard errors by worker id and year. |
| ols | wooldridge::401k | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.3 401(k) participation equation. |
| ols | wooldridge::attend | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 6 Example 6.3 attendance effects on exam score. |
| ols | wooldridge::barium | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.5 barium chloride import equation. |
| ols | wooldridge::bwght | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 5 Example 5.2 birth weight equation. |
| ols | wooldridge::campus | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4 Example 4.4 log-log campus crime equation. |
| ols | wooldridge::ceosal1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 2 Example 2.11 log-log CEO salary equation. |
| ols | wooldridge::ceosal1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 2 Example 2.3 CEO salary on return on equity. |
| ols | wooldridge::consump | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.4 consumption growth on income growth. |
| ols | wooldridge::crime1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.5 arrest records equation with average sentence. |
| ols | wooldridge::crime1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.5 arrest records equation. |
| ols | wooldridge::fertil3 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 13 Example 13.3 fertility distributed lag equation. |
| ols | wooldridge::gpa1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.1 college GPA equation. |
| ols | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4 Section 4.5 log housing price equation. |
| ols | wooldridge::hprice2 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 6 Example 6.2 log housing price equation with rooms quadratic. |
| ols | wooldridge::htv | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 9 Example 9.3 education equation. |
| ols | wooldridge::intdef | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.2 interest rate on inflation and deficit. |
| ols | wooldridge::jtrain | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.3 pooled job training scrap rate equation. |
| ols | wooldridge::kielmc | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 13 Example 13.1 difference-in-differences housing price equation. |
| ols | wooldridge::meap93 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4 math pass rate equation. |
| ols | wooldridge::nyse | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 11 Example 11.4 efficient markets hypothesis. |
| ols | wooldridge::phillips | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.1 static Phillips curve. |
| ols | wooldridge::phillips | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 11 Example 11.5 expectations-augmented Phillips curve. |
| ols | wooldridge::prminwge | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.3 Puerto Rican employment equation. |
| ols | wooldridge::sleep75 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 5 Problem 3.3 sleep equation. |
| ols | wooldridge::twoyear | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4 Example 4.10 returns to college equation. |
| ols | wooldridge::vote1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 2 Examples 2.5 and 2.9 election outcomes equation. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | First real-dataset validation case. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 2 Example 2.10 log wage equation. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 7 Example 7.1 hourly wage equation with female dummy. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 7 Example 7.6 hourly wage equation with marriage-gender interactions. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 7 Example 7.1 log hourly wage equation with female dummy. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 6 Section 6.2 wage equation with experience quadratic. |
| oprobit | wooldridge::beauty | R:passed *, Python:passed * | pass | — | Ordered probit model of self-reported beauty rating (looks 2-5) on female, education, experience and black indicators. |
| panel_fe | wooldridge::wagepan | — | not-supported | 93 | Panel fixed-effects wage equation with worker-clustered standard errors; currently not supported because fe() ignores cluster=. |
| panel_fe | wooldridge::grunfeld | R:passed *, Python:passed * | pass | — | Panel fixed-effects investment demand model (Grunfeld). |
| panel_fe | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 panel fixed-effects wage equation. |
| panel_heckman | simulated_panel_heckman | R:passed *, Python:passed * | pass | — | Panel Heckman selection model (two-step) with selection equation and outcome equation. Uses simulated panel data with known selection mechanism. |
| pca | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Standardised PCA of educ, exper, tenure, and wage; absolute loadings are compared because component signs are arbitrary. |
| pcse | wooldridge::wagepan | Python | pass | 99 | PCSE estimation of log wage on education, experience, and dummies using the Hayashi/Greeners Beck-Katz covariance convention. |
| poisson | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Poisson regression for number of children on the fertil2 dataset. |
| probit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Probit labour-force participation on the Mroz dataset. |
| psm | wooldridge::jtrain3 | R:passed *, Python:passed * | pass | — | 1:1 nearest-neighbor propensity score matching with caliper 0.2 and bootstrap SE. |
| qreg | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Median quantile regression of wage on education, experience, and tenure. |
| rdd | rdd_book | R:passed *, Python:passed * | pass | — | Sharp RDD with local linear regression, triangular kernel and Imbens-Kalyanaraman bandwidth. |
| re | grunfeld | R:passed *, Python:passed * | pass | 101 | Random-effects investment demand model (Grunfeld). |
| ridge | wooldridge::hprice1 | Python:passed * | pass | — | Ridge regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| rlm | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Huber robust linear regression of log wage on education, experience, and tenure. |
| spatial | simulated_spatial_durbin | R:passed *, Python:passed * | pass | — | Spatial Durbin model with spatial lag of dependent variable and spatially lagged independent variables. Uses simulated spatial data with known spatial weights matrix. |
| sur | wooldridge::grunfeld | R:passed *, Python:passed * | pass | — | Two-equation SUR (Zellner FGLS) on the Grunfeld investment data. |
| svar | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | Cholesky-identified SVAR(2) on log US real GDP and consumption. |
| synth | synth_smoking | R:passed *, Python:passed * | pass | — | Synthetic-control ATT on a simulated panel with 10 donors and 1 treated unit. |
| sysgmm | wooldridge::wagepan | Python:passed * | pass | — | System GMM (Blundell-Bond) two-step on Wooldridge wagepan with lags=2. Python reference implements the same two-step System GMM procedure used by Hayashi/Greeners. |
| tobit | wooldridge::mroz | R:passed * | pass | — | Tobit regression of hours worked with left censoring at zero. Hayashi matches AER::tobit at displayed precision; the custom Python MLE is retained as a diagnostic script but is not the active reference. |
| var | simulated_var1 | R:passed *, Python:passed * | pass | — | Uses the same simulated bivariate VAR(1) DGP as Chapter 28 of the book. |
| var | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | VAR(2) on US real GDP and consumption. |
| wls | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | WLS with weights generated inside Hayashi to avoid sandbox file issues. |
| xtgls | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Panel feasible GLS with panel-level heteroskedasticity (Parks/Kmenta, Stata xtgls panels(heteroskedastic)). R and Python references implement the same two-step FGLS procedure used by Hayashi/Greeners. |
| zinb | wooldridge::affairs | R:passed * | pass | — | ZINB model of number of affairs on demographic predictors. |
| zip | wooldridge::affairs | R:passed * | pass | — | ZIP model of number of affairs on demographic predictors. |

## Status legend

- `pass` — Hayashi matches all available references within declared tolerances.
- `partial` — Hayashi matches at least one reference, but other declared references failed or are missing.
- `fail` — Hayashi differs from at least one reference beyond tolerances.
- `blocked` — no declared reference could run; the case cannot be judged.
- `not-supported` — estimator/workflow not supported yet.
- `not-started` — registered but not implemented.

The Reference column shows per-reference status as `name:status`,
where `*` marks the reference used for comparison. A declared
reference that fails or is missing no longer blocks the case; it is
reported in the Reference column while any passing reference is used.

This matrix is generated from `validation/matrix.yml` by `validation/run.py`.

This matrix covers the core empirical estimators. Some commands are
intentionally excluded for the reasons described in the "Estimators not
covered by validation" section of the README.

Esta matriz abrange os estimadores empíricos centrais. Alguns comandos são
deixados de fora intencionalmente pelos motivos descritos na seção
"Estimators not covered by validation" do README.

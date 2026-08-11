# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| ab | wooldridge::grunfeld | R:passed *, Python:passed * | pass | 119 | Arellano-Bond difference GMM for dynamic panel investment demand. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | One-way ANOVA of wage across education groups. |
| arima | simulated_ar1 | R:passed *, Python:passed * | pass | — | Uses the same simulated AR(1) DGP as Chapter 26 of the book. |
| ardl | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | ARDL(1,1) model of US real GDP on consumption. |
| arima | simulated_rw | R:passed *, Python:passed * | pass | — | ARIMA(1,1,0) on a simulated random walk with seed 42. Intercept is excluded from comparison because R/Python references are estimated without trend. |
| arima | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | ARIMA(1,1,1) on log US real GDP via exact Gaussian MLE. |
| arima | simulated_arma11 | R:passed *, Python:passed * | pass | — | Uses the same simulated ARMA(1,1) DGP as Chapter 26 of the book. Intercept is excluded from comparison because Hayashi profiles it out in MLE (SE = 0). |
| autoreg | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | AR(1) on US real GDP with constant and trend. |
| be | simulated | Python:passed * | pass | — | Between estimator on a simulated panel. Entity means are collapsed and an OLS regression is run on N=50 observations. |
| betareg | wooldridge::401k | R:passed *, Python:passed * | pass | 125 | Beta regression on 401k participation rates. Greeners estimates by BFGS with an analytic gradient and matches R betareg. |
| causal_impact | simulated_causal_impact | R:passed *, Python:passed * | pass | — | Bayesian structural time series for counterfactual inference (Brodersen 2015). Uses simulated data with known treatment effect. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Centiles 10, 25, 50, 75, 90 for the wage variable. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | 95% confidence interval for the wage mean. |
| clogit | simulated | R:passed * | pass | — | Simulated matched groups with group fixed effects and a single endogenous regressor. R reference is survival::clogit; groups without within-group variation are dropped at generation time. |
| cloglog | wooldridge::affairs | R:passed *, Python:passed * | pass | — | Complementary log-log GLM on Wooldridge affairs. Fixed cloglog link derivative sign; Hayashi now converges and matches R glm. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Codebook summary for the continuous wage variable. |
| vecm | simulated_cointegrated | R:passed *, Python:passed * | pass | — | VECM(1) on a simulated cointegrated system where y = 2*x + e2 and x = cumsum(e1). Only the cointegration (beta) and adjustment (alpha) coefficients are compared. |
| copula | simulated | R:passed *, Python:passed * | pass | — | Simulated bivariate normal with Pearson correlation 0.6. Hayashi `copula(..., type="gaussian")` returns the empirical correlation matrix, Kendall's tau, and Spearman's rho. The Gaussian copula parameter for a bivariate normal equals the Pearson correlation. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Pairwise correlations of wage, educ, exper, tenure. |
| cox | statsmodels::heart | R:passed *, Python:passed * | pass | — | Cox proportional hazards regression for survival time after heart transplant. |
| cpoisson | simulated | Python:passed * | pass | — | Simulated panel counts with group fixed effects; the reference implements the exact conditional Poisson likelihood (multinomial fixed-total) using analytic gradient and Hessian. |
| dcc_garch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | DCC-GARCH (Dynamic Conditional Correlation GARCH) on NYSE returns. Uses simplified DCC-GARCH(1,1) model. |
| did | wooldridge::kielmc | R:passed *, Python:passed * | pass | — | Difference-in-differences effect of incinerator proximity on log house prices. |
| did | simulated | R:passed *, Python:passed * | pass | — | Simulated 2x2 DiD with ATT=1.5. Interaction coefficient compared against statsmodels OLS with robust standard errors. |
| double_ml | simulated_double_ml | R:passed *, Python:passed * | pass | — | Double Machine Learning (Chernozhukov et al. 2018) for heterogeneous treatment effects. Uses simulated data with known treatment effect. |
| egarch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | EGARCH(1,1) on NYSE returns. |
| elasticnet | wooldridge::hprice1 | Python:passed * | pass | — | Elastic Net regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| ets | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | Exponential smoothing state-space model on US real GDP. Blocked because Hayashi uses SSE grid search while references use MLE. |
| feiv | simulated | R:passed *, Python:passed * | pass | 134 | Panel with N=200 entities and T=5 periods; x is endogenous and instrumented by z. Independent R and Python within-2SLS references use the Greeners residual degrees-of-freedom convention n - k - (G - 1). |
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
| km | survival::aml | R:passed *, Python:passed * | pass | 113 | Kaplan-Meier right-continuous survival probabilities at seven checkpoints on survival::aml. |
| kmeans | simulated_kmeans | R:passed *, Python:passed * | pass | — | K-Means clustering (MacQueen 1967) with k-means++ initialization. Uses simulated 2D data with 3 Gaussian clusters. |
| lasso | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | Lasso regression of house price on lot size, square footage and bedrooms. |
| logit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Logit average marginal effects on Wooldridge mroz. |
| logit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Logit labour-force participation on the Mroz dataset. |
| did | simulated_absorbing_panel | Python:passed * | pass | — | LP-DiD quickstart against pylpdid on an absorbing staggered-adoption panel. R reference is left aside for now. |
| arima | simulated_ma1 | R:passed *, Python:passed * | pass | — | Uses the same simulated MA(1) DGP as Chapter 26 of the book. |
| mice_chained | simulated_mice | R:passed *, Python:passed * | pass | — | MICE (Multiple Imputation by Chained Equations, van Buuren 2011) with m=5, iter=10. Uses simulated data with MCAR missing values. |
| mixed | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 mixed linear model wage equation. |
| mlogit | AER::TravelMode | R:passed *, Python:passed * | pass | — | Multinomial logit of chosen travel mode (air=1, train=2, bus=3, car=4) on income, wait time, vehicle cost and travel time. Alternative-specific attributes are averaged per individual to make them individual-specific covariates. |
| modwt | simulated | R:passed *, Python:passed * | pass | — | Simulated series (trend + 16-period sine + noise). Greeners MODWT uses unnormalised Haar filters, equivalent to `pywt.swt(..., norm=False)`. `pywt` returns coefficients coarse-to-fine; the reference reverses them to match Greeners' W_1 (finest) convention. |
| negbin | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Negative binomial regression for number of children on age, education, electric and urban indicators. Dispersion parameter (alpha) is not compared because Hayashi does not report it; coefficient tolerance is 2e-1 due to different alpha estimates. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a * (b1*x1^rho + (1-b1)*x2^rho)^(1/rho) + N(0, 0.1). The CES function is now identified by the share restriction b2 = 1 - b1. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a * x1^b1 * x2^b2 + N(0, 0.3). Coefficients and standard errors compared against R `nls` and Python `curve_fit`. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Nonlinear least squares exponential model on simulated data. Reference matches y = a * exp(b * x) + N(0, 0.1) against R `nls` and Python `curve_fit`. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a / (1 + exp(-b*(x-c))) + N(0, 0.2). Coefficients and standard errors compared against R `nls` and Python `scipy.optimize.curve_fit`. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a * x^b + N(0, 0.3). Coefficients and standard errors compared against R `nls` and Python `scipy.optimize.curve_fit`. |
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
| panel_fe | wooldridge::wagepan | R:passed *, Python:passed * | pass | 115 | Panel fixed-effects wage equation with worker-clustered standard errors using explicit within-transformed CR1 reference implementations. Tolerance reflects Hayashi's four-decimal text export. |
| panel_fe | wooldridge::grunfeld | R:passed *, Python:passed * | pass | — | Panel fixed-effects investment demand model (Grunfeld). |
| panel_fe | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 panel fixed-effects wage equation. |
| panel_heckman | simulated_panel_heckman | R:passed *, Python:passed * | pass | — | Panel Heckman selection model (two-step) with selection equation and outcome equation. Uses simulated panel data with known selection mechanism. |
| pca | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Standardised PCA of educ, exper, tenure, and wage; absolute loadings are compared because component signs are arbitrary. |
| pcse | wooldridge::wagepan | R:passed *, Python:passed * | pass | 99, 103 | PCSE estimation of log wage on education, experience, and dummies using the Hayashi/Greeners Beck-Katz covariance convention. |
| poisson | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Poisson regression for number of children on the fertil2 dataset. |
| probit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Probit labour-force participation on the Mroz dataset. |
| psm | wooldridge::jtrain3 | R:passed *, Python:passed * | pass | — | 1:1 nearest-neighbor propensity score matching with caliper 0.2 and bootstrap SE. |
| qreg | simulated | R:passed *, Python:passed * | pass | — | Simulated heteroskedastic data y = 1 + 2x + (1 + 0.5x) * (Exponential(1)-1). Quantile regression at tau=0.75 compared against statsmodels. boot=0 to avoid bootstrap overhead. |
| qreg | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Median quantile regression of wage on education, experience, and tenure. |
| rdd | rdd_book | R:passed *, Python:passed * | pass | — | Sharp RDD with local linear regression, triangular kernel and Imbens-Kalyanaraman bandwidth. |
| re | grunfeld | R:passed *, Python:passed * | pass | 101 | Random-effects investment demand model (Grunfeld). |
| rf | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 3*x1 + N(0, 0.1). In-sample R² compared against scikit-learn RandomForestRegressor with the same tree and depth settings. |
| ridge | wooldridge::hprice1 | R:passed *, Python:passed * | pass | 106 | Ridge regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| rlm | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Huber robust linear regression of log wage on education, experience, and tenure. |
| spatial_durbin | simulated | R:passed *, Python:passed * | pass | — | Data generated on a 7x7 grid with rook contiguity W, rho=0.3, beta=0.5, theta=0.2. Only rho is compared because the intercept and spatially lagged intercept are collinear in the cross-sectional specification. |
| spatial_sar | simulated | R:passed *, Python:passed * | pass | — | Spatial autoregressive (SAR) model on a simulated 7x7 grid with rook contiguity weights. Reference implements the same concentrated MLE independently. |
| spatial_sem | simulated | R:passed *, Python:passed * | pass | — | Data generated on a 7x7 grid with rook contiguity W, lambda=0.3, beta=0.5. Reference implements the concentrated MLE for SEM independently. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Summary statistics with detail (percentiles, skewness, kurtosis) for wage. |
| sur | wooldridge::grunfeld | R:passed *, Python:passed * | pass | — | Two-equation SUR (Zellner FGLS) on the Grunfeld investment data. |
| svar | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | Cholesky-identified SVAR(2) on log US real GDP and consumption. |
| svar | simulated | R:passed *, Python:passed * | pass | — | Blanchard-Quah SVAR(1) with long-run restrictions on a simulated bivariate system. Reference implements the same C(1) Cholesky procedure used by Greeners. |
| synth | synth_smoking | R:passed *, Python:passed * | pass | — | Synthetic-control ATT on a simulated panel with 10 donors and 1 treated unit. |
| synthdid | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with 20 units, 10 periods, treatment begins at period 6 for unit 0 with ATT=2.0. Reference uses a simple synthetic-control-style pre-treatment weighting and computes the post-treatment mean gap. |
| sysgmm | wooldridge::wagepan | R:passed *, Python:passed * | pass | 117 | System GMM (Blundell-Bond) two-step on Wooldridge wagepan with lags=2. R and Python references explicitly implement the same two-step System GMM procedure used by Hayashi/Greeners; plm::pgmm is not used as the active R oracle because it uses different instrument and weighting conventions. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Tabstat statistics (mean, sd, min, max, p50) for wage, educ, exper, tenure. |
| descriptive | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Two-way frequency table with Pearson chi-square test. |
| three_sls | simulated | Python:passed * | pass | — | Simultaneous two-equation system with correlated errors. Each equation includes an intercept, one exogenous and one endogenous regressor; the excluded exogenous from the other equation is used as an instrument. Python reference is linearmodels.system.IV3SLS with an explicit constant column. |
| tobit | wooldridge::mroz | R:passed * | pass | — | Tobit regression of hours worked with left censoring at zero. Hayashi matches AER::tobit at displayed precision; the custom Python MLE is retained as a diagnostic script but is not the active reference. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | One-sample t-test of wage mean against mu=5. |
| var | simulated_var1 | R:passed *, Python:passed * | pass | — | Uses the same simulated bivariate VAR(1) DGP as Chapter 28 of the book. |
| var | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | VAR(2) on US real GDP and consumption. |
| varma | simulated | Python:passed * | pass | — | Bivariate VARMA(1,1) with known AR and MA matrices. Hayashi uses the Hannan-Rissanen algorithm; the Python reference uses statsmodels VARMAX with no trend. Coefficients are compared (standard errors are not computed by the current Hayashi VARMA implementation). |
| wls | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | WLS with weights generated inside Hayashi to avoid sandbox file issues. |
| xtgls | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Panel feasible GLS with panel-level heteroskedasticity (Parks/Kmenta, Stata xtgls panels(heteroskedastic)). R and Python references implement the same two-step FGLS procedure used by Hayashi/Greeners. |
| xtlogit | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE logit with exchangeable working correlation. Only coefficients compared; standard errors depend on the sandwich estimator convention used by each package. |
| xtpoisson | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE Poisson with exchangeable working correlation. Only coefficients compared; standard errors depend on the sandwich estimator convention. |
| xtprobit | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE probit with exchangeable working correlation. Only coefficients compared. |
| descriptive | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Overall, between, and within panel summary for lwage. |
| zinb | wooldridge::affairs | R:passed *, Python:passed * | pass | 123 | ZINB model of number of affairs on demographic predictors. |
| zip | wooldridge::affairs | R:passed *, Python:passed * | pass | 121 | ZIP model of number of affairs on demographic predictors. |

## Status legend

- `pass` — Hayashi matches all available references within declared tolerances.
- `partial` — Hayashi matches at least one reference, but other declared references failed or are missing; exits non-zero unless `--allow-partial` is passed.
- `fail` — Hayashi differs from at least one reference beyond tolerances.
- `blocked` — no declared reference could run; the case cannot be judged.
- `not-supported` — estimator/workflow not supported yet.
- `not-started` — registered but not implemented.

The Reference column shows per-reference status as `name:status`,
where `*` marks the reference used for comparison. A declared
reference that fails or is missing no longer blocks comparison when
`--allow-partial` is used; otherwise partial cases fail the runner.

This matrix is generated from `validation/matrix.yml` by `validation/run.py`.

This matrix covers the core empirical estimators. Some commands are
intentionally excluded for the reasons described in the "Estimators not
covered by validation" section of the README.

Esta matriz abrange os estimadores empíricos centrais. Alguns comandos são
deixados de fora intencionalmente pelos motivos descritos na seção
"Estimators not covered by validation" do README.


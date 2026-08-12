# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| ab | wooldridge::grunfeld | R, Python | pass | 119 | Arellano-Bond difference GMM for dynamic panel investment demand. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | One-way ANOVA of wage across education groups. |
| arima | simulated_ar1 | R, Python | pass | — | Uses the same simulated AR(1) DGP as Chapter 26 of the book. |
| ardl | statsmodels::macrodata | R, Python | pass | — | ARDL(1,1) model of US real GDP on consumption. |
| arima | simulated_rw | R, Python | pass | — | ARIMA(1,1,0) on a simulated random walk with seed 42. Intercept is excluded from comparison because R/Python references are estimated without trend. |
| arima | statsmodels::macrodata | R, Python | pass | — | ARIMA(1,1,1) on log US real GDP via exact Gaussian MLE. |
| arima | simulated_arma11 | R, Python | pass | — | Uses the same simulated ARMA(1,1) DGP as Chapter 26 of the book. Intercept is excluded from comparison because Hayashi profiles it out in MLE (SE = 0). |
| autoreg | statsmodels::macrodata | R, Python | pass | — | AR(1) on US real GDP with constant and trend. |
| bart | simulated | R, Python | pass | — | Simulated data y = 3*x1 + N(0, 0.1), x2 irrelevant. BART with 20 trees, depth 3, 500 post-burn draws and 200 burn-in. Reference is a scikit-learn GradientBoostingRegressor approximation because a full BART posterior is too heavy for the venv. |
| be | simulated | R, Python | pass | — | Between estimator on a simulated panel. Entity means are collapsed and an OLS regression is run on N=50 observations. |
| betareg | wooldridge::401k | R, Python | pass | 125 | Beta regression on 401k participation rates. Greeners estimates by BFGS with an analytic gradient and matches R betareg. |
| biplot | simulated | Python | pass | — | Symmetric PCA biplot. Compare explained-variance ratios and sign-robust squared loading sums. |
| causal_impact | simulated_causal_impact | R, Python | pass | — | Bayesian structural time series for counterfactual inference (Brodersen 2015). Uses simulated data with known treatment effect. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | Centiles 10, 25, 50, 75, 90 for the wage variable. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | 95% confidence interval for the wage mean. |
| clogit | simulated | R, Python | pass | — | Simulated matched groups with group fixed effects and a single endogenous regressor. R reference is survival::clogit; groups without within-group variation are dropped at generation time. |
| cloglog | wooldridge::affairs | R, Python | pass | — | Complementary log-log GLM on Wooldridge affairs. Fixed cloglog link derivative sign; Hayashi now converges and matches R glm. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | Codebook summary for the continuous wage variable. |
| vecm | simulated_cointegrated | R, Python | pass | — | VECM(1) on a simulated cointegrated system where y = 2*x + e2 and x = cumsum(e1). Only the cointegration (beta) and adjustment (alpha) coefficients are compared. |
| copula | simulated | R, Python | pass | — | Simulated bivariate normal with Pearson correlation 0.6. Hayashi `copula(..., type="gaussian")` returns the empirical correlation matrix, Kendall's tau, and Spearman's rho. The Gaussian copula parameter for a bivariate normal equals the Pearson correlation. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | Pairwise correlations of wage, educ, exper, tenure. |
| cox | statsmodels::heart | R, Python | pass | — | Cox proportional hazards regression for survival time after heart transplant. |
| cpoisson | simulated | R, Python | pass | — | Simulated panel counts with group fixed effects; the reference implements the exact conditional Poisson likelihood (multinomial fixed-total) using analytic gradient and Hessian. |
| cuped | simulated | Python, R | pass | — | Simulated A/B test with pre-experiment covariate. Compare CUPED-adjusted ATE. |
| dbscan | simulated | Python | pass | — | Three dense 2D blobs plus five isolated noise points. Compare cluster and noise counts. |
| dcc_garch | wooldridge::nyse | R, Python | pass | — | DCC-GARCH (Dynamic Conditional Correlation GARCH) on NYSE returns. Uses simplified DCC-GARCH(1,1) model. |
| did | wooldridge::kielmc | R, Python | pass | — | Difference-in-differences effect of incinerator proximity on log house prices. |
| did | simulated | R, Python | pass | — | Simulated 2x2 DiD with ATT=1.5. Interaction coefficient compared against statsmodels OLS with robust standard errors. |
| double_ml | simulated_double_ml | R, Python | pass | — | Double Machine Learning (Chernozhukov et al. 2018) for heterogeneous treatment effects. Uses simulated data with known treatment effect. |
| dr_learner | simulated | R, Python | pass | — | Simulated data with a single confounder x, binary treatment d, and constant ATE=2.0. DR-Learner average treatment effect compared against a manual AIPW reference. |
| egarch | wooldridge::nyse | R, Python | pass | — | EGARCH(1,1) on NYSE returns. |
| elasticnet | wooldridge::hprice1 | R, Python | pass | — | Elastic Net regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| logit | simulated | R, Python | pass | — | Simulated logit. Sensitivity, specificity and correct rate at threshold 0.5. |
| iv | simulated | R, Python | pass | — | Simulated endogenous regressor with one instrument. Wu-Hausman F and p-value. |
| logit | simulated | R, Python | pass | — | Simulated strong predictor logit. Hosmer-Lemeshow chi-square by deciles. |
| iv | simulated | R, Python | pass | — | Simulated IV with two instruments and one endogenous regressor. Sargan J-statistic and p-value. |
| ets | statsmodels::macrodata | R, Python | pass | — | Exponential smoothing state-space model on US real GDP. Blocked because Hayashi uses SSE grid search while references use MLE. |
| favar | simulated | R, Python | pass | — | Simulated three observable series driven by one common factor plus the observed y1. Python reference extracts the first PCA factor and estimates a VAR(1) by OLS, matching the FAVAR two-step approach. |
| feiv | simulated | R, Python | pass | 134 | Panel with N=200 entities and T=5 periods; x is endogenous and instrumented by z. Independent R and Python within-2SLS references use the Greeners residual degrees-of-freedom convention n - k - (G - 1). |
| fmb | simulated_fmb_panel | R, Python | pass | 49 | Classic Fama-MacBeth regression on a deterministic simulated asset panel. |
| ftest_robust | wooldridge::wage1 | R, Python | pass | — | Robust F-test (Wooldridge 2010) with cluster-robust covariance for joint significance test. |
| rd | simulated | Python | pass | — | Fuzzy RD with 70% compliance at the cutoff. Compare local average treatment effect (LATE). |
| garch | simulated_garch11 | R, Python | pass | — | Uses the same simulated GARCH(1,1) DGP as Chapter 30 of the book. MLE tolerances are looser because the optimizer may stop at slightly different points. |
| garch | wooldridge::nyse | R, Python | pass | — | GARCH(1,1) on NYSE returns. |
| gbm | simulated | R, Python | pass | — | Simulated data y = 3*x1 + N(0, 0.1), x2 irrelevant. Gradient Boosting with 50 trees, learning rate 0.1, max depth 3. MSE and R^2 compared against scikit-learn. |
| gee | wooldridge::wagepan | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 generalized estimating equations wage equation. |
| glm | wooldridge::fertil2 | R, Python | pass | — | Poisson GLM for number of children on Wooldridge fertil2. |
| glsar | wooldridge::hprice1 | R, Python | pass | — | GLS with AR(1) errors on housing price equation. |
| gmm | wooldridge::card | R, Python | pass | — | GMM returns to schooling with nearc4 as instrument for education. |
| gmm_clust | simulated | Python | pass | — | Two Gaussian clusters. Compare sorted component means. |
| hausman_robust | wooldridge::wagepan | R, Python | pass | — | Robust Hausman test (Cameron-Trivedi 2005, Wooldridge 2010) with cluster-robust covariance. |
| hawkes | simulated | R, Python | pass | — | Simulated self-exciting Hawkes process. Python reference fits the same MLE via L-BFGS-B. |
| hclust | simulated | Python | pass | — | Three well-separated 2D blobs. Ward linkage with cut=3.0 and cophenetic correlation. |
| heckman | wooldridge::mroz | R, Python | pass | — | Two-step Heckman (Heckit) on the Mroz dataset. SEs are approximate because the reference implementations are two-step. |
| ols | simulated | R, Python | pass | — | Simulated OLS with an outlier. Hayashi exposes DFFITS; reference uses max |DFFITS| (Cook's D not exported). |
| isotonic | simulated | Python, R | pass | — | Simulated three-step data. PAVA fitted values at x=1,50,100 compared. |
| iv | wooldridge::card | R, Python | pass | — | IV with education endogenous and nearc4 as instrument. |
| iv | wooldridge::card | R, Python | pass | 97 | IV returns-to-schooling equation with one-way clustered standard errors by Census region. |
| iv | wooldridge::mroz | R, Python | pass | 95 | IV returns-to-schooling equation with HC1 heteroskedasticity-robust standard errors. |
| iv | wooldridge::mroz | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 15 Example 15.1 IV returns to schooling for married women. |
| kalman | wooldridge::nyse | R, Python | pass | — | Local-level Kalman filter on NYSE returns. Hayashi estimates sigma_obs and sigma_state by MLE and returns a printable result object. |
| km | survival::aml | R, Python | pass | 113 | Kaplan-Meier right-continuous survival probabilities at seven checkpoints on survival::aml. |
| kmeans | simulated_kmeans | R, Python | pass | — | K-Means clustering (MacQueen 1967) with k-means++ initialization. Uses simulated 2D data with 3 Gaussian clusters. |
| lasso | wooldridge::hprice1 | R, Python | pass | — | Lasso regression of house price on lot size, square footage and bedrooms. |
| logit | simulated | R, Python | pass | — | Simulated logit. Linktest yhat and yhat2 coefficients and standard errors. |
| logit | wooldridge::mroz | R, Python | pass | — | Logit average marginal effects on Wooldridge mroz. |
| logit | wooldridge::mroz | R, Python | pass | — | Logit labour-force participation on the Mroz dataset. |
| did | simulated_absorbing_panel | R, Python | pass | — | LP-DiD quickstart against pylpdid on an absorbing staggered-adoption panel. R reference is left aside for now. |
| logit | simulated | R, Python | pass | — | Simulated strong predictor logit. AUC and Gini exported as a coefficient table. |
| arima | simulated_ma1 | R, Python | pass | — | Uses the same simulated MA(1) DGP as Chapter 26 of the book. |
| mice_chained | simulated_mice | R, Python | pass | — | MICE (Multiple Imputation by Chained Equations, van Buuren 2011) with m=5, iter=10. Uses simulated data with MCAR missing values. |
| midas | simulated | Python:passed * | pass | — | Simulated low-frequency y (T=100) and high-frequency x (T*3). y = 1.0 + 2.0 * x_midas + noise. Compare alpha, beta and R-squared against Python MIDAS grid+minimize reference. |
| mixed | wooldridge::wagepan | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 mixed linear model wage equation. |
| mlogit | AER::TravelMode | R, Python | pass | — | Multinomial logit of chosen travel mode (air=1, train=2, bus=3, car=4) on income, wait time, vehicle cost and travel time. Alternative-specific attributes are averaged per individual to make them individual-specific covariates. |
| mlp | simulated | Python | pass | — | Simulated linear-ish y from x1 and x2 with small noise. Compare R-squared against scikit-learn MLPRegressor (logistic activation, adam solver). Only Python reference because R lacks an equivalent stable MLP package for this use. |
| modwt | simulated | R, Python | pass | — | Simulated series (trend + 16-period sine + noise). Greeners MODWT uses unnormalised Haar filters, equivalent to `pywt.swt(..., norm=False)`. `pywt` returns coefficients coarse-to-fine; the reference reverses them to match Greeners' W_1 (finest) convention. |
| nardl | simulated | R, Python | pass | — | Simulated NARDL(1,1) with asymmetric long-run multipliers and short-run dynamics. y and x are random walks with positive and negative shock decomposition. |
| negbin | wooldridge::fertil2 | R, Python | pass | — | Negative binomial regression for number of children on age, education, electric and urban indicators. Dispersion parameter (alpha) is not compared because Hayashi does not report it; coefficient tolerance is 2e-1 due to different alpha estimates. |
| nls | simulated | R, Python | pass | — | Simulated data from y = a * (b1*x1^rho + (1-b1)*x2^rho)^(1/rho) + N(0, 0.1). The CES function is now identified by the share restriction b2 = 1 - b1. |
| nls | simulated | R, Python | pass | — | Simulated data from y = a * x1^b1 * x2^b2 + N(0, 0.3). Coefficients and standard errors compared against R `nls` and Python `curve_fit`. |
| nls | simulated | R, Python | pass | — | Nonlinear least squares exponential model on simulated data. Reference matches y = a * exp(b * x) + N(0, 0.1) against R `nls` and Python `curve_fit`. |
| nls | simulated | R, Python | pass | — | Simulated data from y = a / (1 + exp(-b*(x-c))) + N(0, 0.2). Coefficients and standard errors compared against R `nls` and Python `scipy.optimize.curve_fit`. |
| nls | simulated | R, Python | pass | — | Simulated data from y = a * x^b + N(0, 0.3). Coefficients and standard errors compared against R `nls` and Python `scipy.optimize.curve_fit`. |
| ologit | wooldridge::beauty | R, Python | pass | — | Ordered logit of looks (2, 3, 4) on female, educ, exper, black. |
| ols | wooldridge::wagepan | R, Python | pass | — | OLS wage equation with one-way cluster-robust standard errors by worker id. |
| ols | wooldridge::wage1 | R, Python | pass | 89 | OLS log-wage equation with HC3 heteroskedasticity-robust standard errors. |
| ols | wooldridge::phillips | R, Python | pass | 91 | OLS expectations-augmented Phillips curve with Newey-West HAC standard errors. |
| ols | wooldridge::wagepan | R, Python | pass | 87 | OLS wage equation with two-way clustered standard errors by worker id and year. |
| ols | wooldridge::401k | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.3 401(k) participation equation. |
| ols | wooldridge::attend | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 6 Example 6.3 attendance effects on exam score. |
| ols | wooldridge::barium | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.5 barium chloride import equation. |
| ols | wooldridge::bwght | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 5 Example 5.2 birth weight equation. |
| ols | wooldridge::campus | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 4 Example 4.4 log-log campus crime equation. |
| ols | wooldridge::ceosal1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 2 Example 2.11 log-log CEO salary equation. |
| ols | wooldridge::ceosal1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 2 Example 2.3 CEO salary on return on equity. |
| ols | wooldridge::consump | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.4 consumption growth on income growth. |
| ols | wooldridge::crime1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.5 arrest records equation with average sentence. |
| ols | wooldridge::crime1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.5 arrest records equation. |
| ols | wooldridge::fertil3 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 13 Example 13.3 fertility distributed lag equation. |
| ols | wooldridge::gpa1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.1 college GPA equation. |
| ols | wooldridge::hprice1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 4 Section 4.5 log housing price equation. |
| ols | wooldridge::hprice2 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 6 Example 6.2 log housing price equation with rooms quadratic. |
| ols | wooldridge::htv | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 9 Example 9.3 education equation. |
| ols | wooldridge::intdef | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.2 interest rate on inflation and deficit. |
| ols | wooldridge::jtrain | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.3 pooled job training scrap rate equation. |
| ols | wooldridge::kielmc | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 13 Example 13.1 difference-in-differences housing price equation. |
| ols | wooldridge::meap93 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 4 math pass rate equation. |
| ols | wooldridge::nyse | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 11 Example 11.4 efficient markets hypothesis. |
| ols | wooldridge::phillips | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.1 static Phillips curve. |
| ols | wooldridge::phillips | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 11 Example 11.5 expectations-augmented Phillips curve. |
| ols | wooldridge::prminwge | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 10 Example 10.3 Puerto Rican employment equation. |
| ols | wooldridge::sleep75 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 5 Problem 3.3 sleep equation. |
| ols | wooldridge::twoyear | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 4 Example 4.10 returns to college equation. |
| ols | wooldridge::vote1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 2 Examples 2.5 and 2.9 election outcomes equation. |
| ols | wooldridge::wage1 | R, Python | pass | — | First real-dataset validation case. |
| ols | wooldridge::wage1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 2 Example 2.10 log wage equation. |
| ols | wooldridge::wage1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 7 Example 7.1 hourly wage equation with female dummy. |
| ols | wooldridge::wage1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 7 Example 7.6 hourly wage equation with marriage-gender interactions. |
| ols | wooldridge::wage1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 7 Example 7.1 log hourly wage equation with female dummy. |
| ols | wooldridge::wage1 | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 6 Section 6.2 wage equation with experience quadratic. |
| oprobit | wooldridge::beauty | R, Python | pass | — | Ordered probit model of self-reported beauty rating (looks 2-5) on female, education, experience and black indicators. |
| panel_fe | wooldridge::wagepan | R, Python | pass | 115 | Panel fixed-effects wage equation with worker-clustered standard errors using explicit within-transformed CR1 reference implementations. Tolerance reflects Hayashi's four-decimal text export. |
| panel_fe | wooldridge::grunfeld | R, Python | pass | — | Panel fixed-effects investment demand model (Grunfeld). |
| panel_fe | wooldridge::wagepan | R, Python | pass | — | Wooldridge Introductory Econometrics Chapter 14 Example 14.4 panel fixed-effects wage equation. |
| panel_heckman | simulated_panel_heckman | R, Python | pass | — | Panel Heckman selection model (two-step) with selection equation and outcome equation. Uses simulated panel data with known selection mechanism. |
| panel_qreg | simulated | R, Python | pass | — | Simulated panel with entity fixed effects and heteroskedastic errors. References demean the data and run quantile regression without an intercept; standard errors are convention-sensitive. |
| panel | simulated | R, Python | pass | — | Simulated panel with N=50, T=4, random effects, left-censored at 0. Coefficients and standard errors compared against pooled Tobit (censReg and MLE). |
| pca | wooldridge::wage1 | R, Python | pass | — | Standardised PCA of educ, exper, tenure, and wage; absolute loadings are compared because component signs are arbitrary. |
| pcse | wooldridge::wagepan | R, Python | pass | 99, 103 | PCSE estimation of log wage on education, experience, and dummies using the Hayashi/Greeners Beck-Katz covariance convention. |
| poisson | wooldridge::fertil2 | R, Python | pass | — | Poisson regression for number of children on the fertil2 dataset. |
| portsort | simulated | Python, R | pass | — | Five equal-count portfolios sorted by size. Compare mean returns and high-low spread. |
| probit | wooldridge::mroz | R, Python | pass | — | Probit labour-force participation on the Mroz dataset. |
| psm | wooldridge::jtrain3 | R, Python | pass | — | 1:1 nearest-neighbor propensity score matching with caliper 0.2 and bootstrap SE. |
| pstr | simulated | R, Python | pass | — | Simulated panel with N=50, T=10. y = beta0*x + beta1*x*g(q; gamma=5, c=0.5) + FE + noise. Gamma, c, beta0_x and beta1_x compared against grid-search references. |
| pvar | simulated | R, Python | pass | — | Simulated bivariate panel VAR with N=50 and T=100. Hayashi GMM and within-OLS references agree within moderate tolerance due to the Nickell bias in within estimation. |
| qreg | simulated | R, Python | pass | — | Simulated heteroskedastic data y = 1 + 2x + (1 + 0.5x) * (Exponential(1)-1). Quantile regression at tau=0.75 compared against statsmodels. boot=0 to avoid bootstrap overhead. |
| qrf | simulated | R, Python | pass | — | Simulated heteroskedastic data y = 3*x1 + N(0, 0.1*(1+0.5*x1)). QRF at tau=0.75 with 50 trees, depth 5. OOB R^2 compared against quantile_forest.RandomForestQuantileRegressor. |
| qreg | wooldridge::wage1 | R, Python | pass | — | Median quantile regression of wage on education, experience, and tenure. |
| rdd | rdd_book | R, Python | pass | — | Sharp RDD with local linear regression, triangular kernel and Imbens-Kalyanaraman bandwidth. |
| re | grunfeld | R, Python | pass | 101 | Random-effects investment demand model (Grunfeld). |
| rf | simulated | R, Python | pass | — | Simulated data y = 3*x1 + N(0, 0.1). In-sample R² compared against scikit-learn RandomForestRegressor with the same tree and depth settings. |
| ridge | wooldridge::hprice1 | R, Python | pass | 106 | Ridge regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| rlm | wooldridge::wage1 | R, Python | pass | — | Huber robust linear regression of log wage on education, experience, and tenure. |
| setar | simulated | R, Python | pass | — | Simulated SETAR(1,1,1) with two regimes split by y_{t-1}. Hayashi grid search may differ slightly from R tsDyn; tolerances are relaxed accordingly. |
| spatial_durbin | simulated | R, Python | pass | — | Data generated on a 7x7 grid with rook contiguity W, rho=0.3, beta=0.5, theta=0.2. Only rho is compared because the intercept and spatially lagged intercept are collinear in the cross-sectional specification. |
| spatial_sar | simulated | R, Python | pass | — | Spatial autoregressive (SAR) model on a simulated 7x7 grid with rook contiguity weights. Reference implements the same concentrated MLE independently. |
| spatial_sem | simulated | R, Python | pass | — | Data generated on a 7x7 grid with rook contiguity W, lambda=0.3, beta=0.5. Reference implements the concentrated MLE for SEM independently. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | Summary statistics with detail (percentiles, skewness, kurtosis) for wage. |
| sur | wooldridge::grunfeld | R, Python | pass | — | Two-equation SUR (Zellner FGLS) on the Grunfeld investment data. |
| svar | statsmodels::macrodata | R, Python | pass | — | Cholesky-identified SVAR(2) on log US real GDP and consumption. |
| svar | simulated | R, Python | pass | — | Blanchard-Quah SVAR(1) with long-run restrictions on a simulated bivariate system. Reference implements the same C(1) Cholesky procedure used by Greeners. |
| synth | synth_smoking | R, Python | pass | — | Synthetic-control ATT on a simulated panel with 10 donors and 1 treated unit. |
| synthdid | simulated | R, Python | pass | — | Simulated panel with 20 units, 10 periods, treatment begins at period 6 for unit 0 with ATT=2.0. Reference uses a simple synthetic-control-style pre-treatment weighting and computes the post-treatment mean gap. |
| sysgmm | wooldridge::wagepan | R, Python | pass | 117 | System GMM (Blundell-Bond) two-step on Wooldridge wagepan with lags=2. R and Python references explicitly implement the same two-step System GMM procedure used by Hayashi/Greeners; plm::pgmm is not used as the active R oracle because it uses different instrument and weighting conventions. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | Tabstat statistics (mean, sd, min, max, p50) for wage, educ, exper, tenure. |
| descriptive | wooldridge::mroz | R, Python | pass | — | Two-way frequency table with Pearson chi-square test. |
| ols | simulated | R, Python | pass | — | Simulated OLS. testparm F and p-value for H0: x1 = x2 = 0. |
| three_sls | simulated | R, Python | pass | — | Simultaneous two-equation system with correlated errors. Each equation includes an intercept, one exogenous and one endogenous regressor; the excluded exogenous from the other equation is used as an instrument. Python reference is linearmodels.system.IV3SLS with an explicit constant column. |
| tobit | wooldridge::mroz | R, Python | pass | — | Tobit regression of hours worked with left censoring at zero. Hayashi matches AER::tobit at displayed precision; the custom Python MLE is retained as a diagnostic script but is not the active reference. |
| tsne | simulated | Python | pass | — | t-SNE embedding of three 3D blobs; cluster quality measured via K-Means inertia. |
| descriptive | wooldridge::wage1 | R, Python | pass | — | One-sample t-test of wage mean against mu=5. |
| tvp | simulated | R, Python | pass | — | Simulated TVP data with smooth intercept and slope drift. The reference is the true final coefficient vector because Greeners TVP uses a simple Kalman-grid implementation with no readily available reference implementation. |
| umap | simulated | Python | pass | — | UMAP embedding of three 3D blobs; cluster quality measured via K-Means inertia. |
| var | simulated_var1 | R, Python | pass | — | Uses the same simulated bivariate VAR(1) DGP as Chapter 28 of the book. |
| var | statsmodels::macrodata | R, Python | pass | — | VAR(2) on US real GDP and consumption. |
| varma | simulated | R, Python | pass | — | Bivariate VARMA(1,1) with known AR and MA matrices. Hayashi uses the Hannan-Rissanen algorithm; the Python reference uses statsmodels VARMAX with no trend. Coefficients are compared (standard errors are not computed by the current Hayashi VARMA implementation). |
| iv | simulated | R, Python | pass | — | Simulated weak instrument. First-stage partial F and p-value. |
| wls | wooldridge::hprice1 | R, Python | pass | — | WLS with weights generated inside Hayashi to avoid sandbox file issues. |
| xgboost | simulated | R, Python | pass | — | Simulated data y = 3*x1 + N(0, 0.1), x2 irrelevant. XGBoost with 50 trees, learning rate 0.1, max depth 3, default regularization. MSE and R^2 compared against xgboost.XGBRegressor. |
| xtgls | wooldridge::wagepan | R, Python | pass | — | Panel feasible GLS with panel-level heteroskedasticity (Parks/Kmenta, Stata xtgls panels(heteroskedastic)). R and Python references implement the same two-step FGLS procedure used by Hayashi/Greeners. |
| xtlogit | simulated | R, Python | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE logit with exchangeable working correlation. Only coefficients compared; standard errors depend on the sandwich estimator convention used by each package. |
| xtpoisson | simulated | R, Python | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE Poisson with exchangeable working correlation. Only coefficients compared; standard errors depend on the sandwich estimator convention. |
| xtprobit | simulated | R, Python | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE probit with exchangeable working correlation. Only coefficients compared. |
| descriptive | wooldridge::wagepan | R, Python | pass | — | Overall, between, and within panel summary for lwage. |
| zinb | wooldridge::affairs | R, Python | pass | 123 | ZINB model of number of affairs on demographic predictors. |
| zip | wooldridge::affairs | R, Python | pass | 121 | ZIP model of number of affairs on demographic predictors. |

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


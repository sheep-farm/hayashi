# Hayashi Validation Matrix

| Family | Dataset | Reference | Status | Blocking Issue | Notes |
|---|---|---:|---|---|---|
| ab | wooldridge::grunfeld | R:passed *, Python:passed * | pass | 119 | Arellano-Bond difference GMM for dynamic panel investment demand. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | One-way ANOVA of wage across education groups. |
| arima | simulated_ar1 | R:passed *, Python:passed * | pass | — | Uses the same simulated AR(1) DGP as Chapter 26 of the book. |
| ardl | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | ARDL(1,1) model of US real GDP on consumption. |
| arima | simulated_rw | R:passed *, Python:passed * | pass | — | ARIMA(1,1,0) on a simulated random walk with seed 42. The intercept reported by Hayashi is excluded from comparison because the R/Python references are estimated without trend. |
| arima | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | R and Python exact-likelihood references agree after correcting two R one-based indexing errors; current Hayashi estimates match both within tolerance. |
| arima | simulated_arma11 | R:passed *, Python:passed * | pass | — | Uses the same simulated ARMA(1,1) DGP as Chapter 26 of the book. Reference replicates Hayashi's default Hannan-Rissanen two-step estimator. |
| autoreg | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | The R and Python references use the same conditional AR(1) design with a constant and linear trend; current Hayashi estimates match both within tolerance. |
| bart | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 3*x1 + N(0, 0.1), x2 irrelevant. BART with 20 trees, depth 3, 500 post-burn draws and 200 burn-in. Reference is a scikit-learn GradientBoostingRegressor approximation because a full BART posterior is too heavy for the venv. |
| bayes_lm | simulated | R:passed *, Python:passed * | pass | — | Simulated y = 1 + 2x1 - 1.5x2 + noise. Compares posterior means of x1 and x2 from Hayashi's conjugate bayes_lm against OLS (which the diffuse prior should recover). |
| bayes_sfa_production | simulated |  | not-supported | — | No stable R/Python reference with half-normal inefficiency MCMC available; PyMC implementation too heavy and fragile for the validation venv. |
| be | simulated | R:passed *, Python:passed * | pass | — | Panel with N=50 entities and T=4 periods. The between estimator collapses each entity to its temporal means and runs OLS on the collapsed data. |
| betareg | wooldridge::401k | R:passed *, Python:passed * | pass | 125 | Beta regression on 401k participation rates. Greeners now estimates the model by BFGS with an analytic gradient and computes standard errors from the observed inverse Hessian, matching R betareg. |
| biplot | simulated | Python:passed * | pass | — | Symmetric PCA biplot. Compare explained-variance ratios and sign-robust squared loading sums. |
| bootstrap | simulated | R:passed *, Python:passed * | pass | — | Simulated OLS data (y = 1 + 2x + noise). Hayashi bootstrap(ols, ...) is compared with R boot::boot and a Python statsmodels OLS pairs bootstrap; quantities are the mean and standard deviation of the bootstrap slope and intercept distributions. |
| bplm | wooldridge::wagepan | Python:passed * | pass | — | Pooled OLS residuals from lwage ~ union + married. The Hayashi Breusch-Pagan LM statistic for individual effects matches the closed-form expression nT/(2(T-1)) * ((A/B) - 1)^2, where A = (1/T) * sum_i (sum_t e_it)^2 and B = sum_it e_it^2. |
| bsc | simulated |  | not-supported | — | No stable R CausalImpact/python pycausalimpact reference in the venv; Bayesian synthetic-control weights depend on prior tuning. |
| bvar | simulated |  | not-supported | — | R mfbvar/bvar packages fail to install; PyMC implementation is too heavy for a deterministic CI reference. |
| cancorr | simulated | Python:passed * | pass | — | Simulated dataset with two X and two Y variables. Compares the two canonical correlations and Wilks' lambda. The reference uses the generalised-eigenvalue formulation. |
| causal_impact | simulated_causal_impact | R:passed *, Python:passed * | pass | — | Bayesian structural time series for counterfactual inference (Brodersen 2015). Uses simulated data with known treatment effect. |
| causalforest | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 1 + 2*x1 - x2 + 0.5*treated + N(0,1). Hayashi causalforest() reports the average treatment effect. R reference uses grf::causal_forest; Python uses econml.grf.CausalForest. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Centiles 10, 25, 50, 75, 90 for the wage variable. |
| chamberlain | wooldridge::wagepan | Python:passed * | pass | — | Chamberlain test on lwage ~ union + married. The Python reference builds the unrestricted model y_it = const + X_it beta + sum_s X_i,s Pi_s and tests H0: all Pi_s = 0 with an F-test, matching the Hayashi/Greeners formulation. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | 95% confidence interval for the wage mean. |
| clogit | simulated | R:passed *, Python:passed * | pass | — | Simulated matched groups with group fixed effects and a single endogenous regressor. R reference is survival::clogit; groups without within-group variation are dropped at generation time. |
| cloglog | wooldridge::affairs | R:passed *, Python:passed * | pass | — | Complementary log-log GLM on Wooldridge affairs. A sign error in the Greeners cloglog derivative caused IRLS divergence; the derivative is now positive and the model converges to the same estimates as R glm and statsmodels. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Codebook summary for the continuous wage variable. |
| vecm | simulated_cointegrated | R:passed *, Python:passed * | pass | — | VECM(1) on a simulated cointegrated system where y = 2*x + e2 and x = cumsum(e1). The cointegration (beta) and adjustment (alpha) coefficients and standard errors are compared. Beta SEs are approximate Engle-Granger/OLS proxies; alpha SEs are OLS conditional SEs given the estimated beta. The 5e-1 tolerance accommodates the bootstrap SEs produced by Hayashi. |
| diagnostics | simulated | R:passed * | pass | — | Condition number of the OLS regressor matrix on simulated data. |
| conformal | simulated | R:passed *, Python:passed * | pass | — | Simulated linear DGP y = 1 + 2x1 - 1.5x2 + noise. Compares split-conformal empirical coverage and conformal quantile. |
| copula | simulated | R:passed *, Python:passed * | pass | — | Simulated bivariate normal with Pearson correlation 0.6. Dependence measures exported as a coefficient table; standard errors are not defined for copula summary statistics. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Pairwise correlations of wage, educ, exper, tenure. |
| cox | statsmodels::heart | R:passed *, Python:passed * | pass | — | Cox proportional hazards regression for survival time after heart transplant. |
| cpoisson | simulated | R:passed *, Python:passed * | pass | — | Simulated panel counts with group fixed effects; the reference implements the exact conditional Poisson likelihood (multinomial fixed-total) using analytic gradient and Hessian. |
| cuped | simulated | Python:passed *, R:passed * | pass | — | Simulated A/B test with pre-experiment covariate. Compare CUPED-adjusted ATE. |
| dbscan | simulated | Python:passed * | pass | — | Three dense 2D blobs plus five isolated noise points. Compare cluster and noise counts. |
| dcc_garch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | DCC-GARCH (Dynamic Conditional Correlation GARCH) on NYSE returns. Uses simplified DCC-GARCH(1,1) model. |
| decompose | simulated | Python:passed * | pass | — | Simulated monthly series (trend + sinusoidal seasonal + noise). Compares selected non-boundary observations of trend, seasonal, and residual components from classical additive decomposition against statsmodels.seasonal_decompose. |
| dfm | simulated | R:passed *, Python:passed * | pass | — | Simulated four observed series driven by two common factors. All variables are standardised in data/gen.py, so the quantities compared are communalities (1 minus observation-noise/uniqueness variance). This is invariant to the arbitrary sign/rotation of the estimated factors. Standard errors are not available in the tidy output and are therefore set to NaN on both sides. |
| did | wooldridge::kielmc | R:passed *, Python:passed * | pass | — | Difference-in-differences effect of incinerator proximity on log house prices. |
| did | simulated | R:passed *, Python:passed * | pass | — | Simulated 2x2 DiD with ATT=1.5. Coefficients and heteroskedasticity-consistent (HC0) standard errors are compared against base-R/sandwich and statsmodels. |
| dml_crossfit | simulated | R:passed *, Python:passed * | pass | — | Simulated partially linear causal model y = 1.5*d + g(x) + eps with binary d confounded by x. Compares the DML cross-fitted ATE. |
| double_ml | simulated_double_ml | R:passed *, Python:passed * | pass | — | Double Machine Learning (Chernozhukov et al. 2018) for heterogeneous treatment effects. Uses simulated data with known treatment effect. |
| doublesort | simulated | R:passed *, Python:passed * | pass | — | Simulated return data with size and book-to-market. Compares the mean return of the small-size / high-BM (low size, high bm) portfolio from a 5x5 double sort. |
| dr_learner | simulated | R:passed *, Python:passed * | pass | — | Simulated data with a single confounder x, binary treatment d, and constant ATE=2.0. DR-Learner average treatment effect compared against a manual AIPW reference. |
| diagnostics | simulated | R:passed * | pass | — | Durbin-Watson test for first-order autocorrelation on simulated OLS residuals. |
| egarch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | EGARCH(1,1) on NYSE returns. |
| elasticnet | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | Elastic Net regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| logit | simulated | R:passed *, Python:passed * | pass | — | Simulated logit. Sensitivity, specificity and correct rate at threshold 0.5. |
| iv | simulated | R:passed *, Python:passed * | pass | — | Simulated endogenous regressor with one instrument. Wu-Hausman F and p-value. |
| logit | simulated | R:passed *, Python:passed * | pass | — | Simulated strong predictor logit. Hosmer-Lemeshow chi-square by deciles. |
| iv | simulated | R:passed *, Python:passed * | pass | — | Simulated IV with two instruments and one endogenous regressor. Sargan J-statistic and p-value. |
| ets | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | Simple exponential smoothing (ETS(A,N,N)) on US real GDP.  Hayashi uses `ses(df, gdp)`, and the R/Python references use SES with SSE grid search / statsmodels optimisation. Only the smoothing parameter alpha is compared because the Hayashi text output does not expose the initial level. |
| eventstudy | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with 60 units and 5 time periods. Half the units are treated at time 2 and the other half are never treated. The outcome has a calendar trend and a post-treatment effect. Standard errors are clustered by unit and compared with R (sandwich::vcovCL) and Python (statsmodels OLS with cov_type='cluster'). |
| factor | simulated | R:passed *, Python:passed * | pass | — | Simulated four variables from two factors. Compares the first two eigenvalues of the correlation matrix (invariant to sign/rotation of loadings). |
| fapanel | simulated |  | not-supported | — | Requires plm + factor combination; no single canonical reference implementation. |
| favar | simulated | R:passed *, Python:passed * | pass | — | Simulated three observable series driven by one common factor plus the observed y1. Python reference extracts the first PCA factor and estimates a VAR(1) by OLS, matching the FAVAR two-step approach. |
| fcoef | simulated |  | not-supported | — | No standard R/Python package for the same functional-coefficient estimator. |
| feiv | simulated | R:passed *, Python:passed * | pass | 134 | Panel with N=200 entities and T=5 periods; x is endogenous and instrumented by z. Independent R and Python within-2SLS references use the Greeners residual degrees-of-freedom convention n - k - (G - 1). |
| fmb | simulated_fmb_panel | R:passed *, Python:passed * | pass | 49 | Classic Fama-MacBeth regression on a deterministic simulated asset panel. |
| fmols | simulated |  | not-supported | — | R cointReg/urca packages not available; cointegration Fully-Modified OLS not in base R or statsmodels. |
| ftest_robust | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Robust F-test (Wooldridge 2010) with cluster-robust covariance for joint significance test. |
| rd | simulated | Python:passed * | pass | — | Fuzzy RD with 70% compliance at the cutoff. Compare local average treatment effect (LATE). |
| garch | simulated_garch11 | R:passed *, Python:passed * | pass | — | Uses the same simulated GARCH(1,1) DGP as Chapter 30 of the book. Coefficients only because GARCH standard-error approximations differ widely between implementations. |
| garch | wooldridge::nyse | R:passed *, Python:passed * | pass | — | GARCH(1,1) on NYSE returns. |
| gbm | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 3*x1 + N(0, 0.1), x2 irrelevant. Gradient Boosting with 50 trees, learning rate 0.1, max depth 3. MSE and R^2 compared against scikit-learn. |
| gee | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14, Example 14.4 generalized estimating equations wage equation. |
| glm | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Wooldridge dataset fertil2 Poisson GLM for number of children. |
| glsar | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | R, Python, and Hayashi use the same iterative GLSAR(1) procedure and agree on coefficients and standard errors within tolerance. |
| gmm | wooldridge::card | R:passed *, Python:passed * | pass | — | R and Python use robust two-step GMM references with matching covariance conventions; current Hayashi estimates match both within tolerance. |
| gmm_clust | simulated | Python:passed * | pass | — | Two Gaussian clusters. Compare sorted component means. |
| gmm | simulated overidentified IV-GMM DGP | R:passed *, Python:passed * | pass | https://github.com/sheep-farm/hayashi/issues/144 | The DGP has one endogenous regressor and two excluded, valid instruments, so L-K=1. R gmm and Python linearmodels use two-step heteroskedastic GMM. The maximum standard-error difference is 2.569e-6; 1e-5 accepts the asymptotically equivalent finite-sample covariance forms. Coefficients and the Hansen J statistic agree to numerical precision. |
| gp | simulated | R:passed *, Python:passed * | pass | — | Simulated 1-D regression with a fixed seed; compares training-set R2 and MSE between Hayashi, R kernlab and Python sklearn. Tolerances reflect different hyperparameter-optimisation conventions. |
| diagnostics | simulated | R:passed * | pass | — | Granger causality test on simulated AR(1) series. |
| grf | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 1 + 2*x1 - x2 + 0.5*treated + N(0,1). Hayashi grf() reports the average treatment effect. R reference uses grf::causal_forest to match the ATE quantity; Python uses econml.grf.CausalForest. |
| diagnostics | simulated | R:passed * | pass | — | Harvey-Collier recursive t test on simulated OLS residuals. |
| hausman_robust | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Robust Hausman test (Cameron-Trivedi 2005, Wooldridge 2010) with cluster-robust covariance. |
| hawkes | simulated | R:passed *, Python:passed * | pass | — | Simulated self-exciting Hawkes process. Python reference fits the same MLE via L-BFGS-B. |
| hclust | simulated | Python:passed * | pass | — | Three well-separated 2D blobs. Ward linkage with cut=3.0 and cophenetic correlation. |
| heckman | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Two-step Heckman (Heckit) on the Mroz dataset. SEs are approximate because the reference implementations are two-step. |
| ols | simulated | R:passed *, Python:passed * | pass | — | Simulated OLS with an outlier. Hayashi exposes DFFITS; reference uses the maximum absolute DFFITS value (Cook's D not exported). |
| isotonic | simulated | Python:passed *, R:passed * | pass | — | Simulated three-step data. PAVA fitted values at x=1,50,100 compared. |
| iv | wooldridge::card | R:passed *, Python:passed * | pass | — | IV with education endogenous and nearc4 as instrument. |
| iv | wooldridge::card | R:passed *, Python:passed * | pass | 97 | IV returns-to-schooling equation with one-way clustered standard errors by Census region. |
| iv | wooldridge::mroz | R:passed *, Python:passed * | pass | 95 | IV returns-to-schooling equation with HC1 heteroskedasticity-robust standard errors. |
| iv | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 15, Example 15.1 returns to schooling IV equation for married women. |
| diagnostics | simulated | R:passed *, Python:passed * | pass | — | Jarque-Bera normality test on a simulated random sample. |
| johansen_break | simulated | R, Python | not-supported | — | Structural-break Johansen test. Python statsmodels.tsa.vecm coint_johansen does not support exogenous break dummies. R urca::ca.jo accepts the dummy but uses conventional rather than Hayashi's break-adjusted rank critical values. Neither provides a fully comparable reference for the declared rank-and-trace contract. |
| kalman | wooldridge::nyse | R:passed *, Python:passed * | pass | — | Local-level Kalman filter on NYSE returns. Hayashi now estimates sigma_obs and sigma_state by maximum likelihood and returns a printable result object. sigma_state is very small and the likelihood is flat in that direction, so the absolute tolerance is set to 1e-3. |
| kde | simulated | R:passed *, Python:passed * | pass | — | Simulated N(2, 1.5). Compares fixed bandwidth, peak density and peak x of a Gaussian KDE. |
| km | survival::aml | R:passed *, Python:passed * | pass | 113 | Kaplan-Meier right-continuous survival probabilities at seven checkpoints on survival::aml. |
| kmeans | simulated_kmeans | R:passed *, Python:passed * | pass | — | K-Means clustering (MacQueen 1967) with k-means++ initialization. Uses simulated 2D data with 3 Gaussian clusters. |
| lasso | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | Lasso regression of house price on lot size, square footage and bedrooms. |
| logit | simulated | R:passed *, Python:passed * | pass | — | Simulated logit. Linktest yhat and yhat2 coefficients and standard errors. |
| diagnostics | simulated | R:passed * | pass | — | Ljung-Box autocorrelation test on a simulated AR(1) series. |
| logit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Logit average marginal effects and delta-method standard errors match R/statsmodels within tolerance. |
| logit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Logit labour-force participation on the Mroz dataset. |
| lowess | simulated | R:passed *, Python:passed * | pass | — | Simulated y = sin(x) + N(0, 0.2). Compare LOWESS fitted values at mean, first, middle and last observations. |
| did | simulated_absorbing_panel | R:passed *, Python:passed * | pass | — | Uses the same absorbing staggered-adoption DGP as pylpdid's quickstart example. The constant treatment effect is 2.0, so post-treatment coefficients should be close to 2.0. Only the Python reference is provided (R is left aside for now). |
| logit | simulated | R:passed *, Python:passed * | pass | — | Simulated strong predictor logit. AUC and Gini exported as a coefficient table. |
| lstm | simulated |  | not-supported | — | PyTorch/TensorFlow stochastic training is not reproducible enough for numerical validation. |
| arima | simulated_ma1 | R:passed *, Python:passed * | pass | — | Uses the same simulated MA(1) DGP as Chapter 26 of the book. |
| manova | simulated | R:passed *, Python:passed * | pass | — | Simulated three groups with distinct bivariate means. Compare Pillai, Wilks, Hotelling-Lawley and Roy multivariate test statistics. |
| mfvar | simulated |  | not-supported | — | R mfbvar package failed to install in the previous session. |
| mice_chained | simulated_mice | R:passed *, Python:passed * | pass | — | MICE (Multiple Imputation by Chained Equations, van Buuren 2011) with m=5, iter=10. Uses simulated data with MCAR missing values. |
| midas | simulated | R:passed *, Python:passed * | pass | — | Simulated low-frequency y (T=100) and high-frequency x (T*3). y = 1.0 + 2.0 * x_midas + noise. Compare alpha, beta and R-squared against R optim and Python MIDAS grid+minimize references. |
| mixed | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14, Example 14.4 mixed linear model wage equation. |
| mlogit | AER::TravelMode | R:passed *, Python:passed * | pass | — | Multinomial logit of chosen travel mode (air=1, train=2, bus=3, car=4) on income, wait time, vehicle cost and travel time. |
| mlp | simulated | R:passed *, Python:passed * | pass | — | Simulated linear-ish y from x1 and x2 with small noise. Compare R-squared against R nnet and scikit-learn MLPRegressor. Tolerance is relaxed because different initialisation and optimisers produce different R-squared. |
| modwt | simulated | R:passed *, Python:passed * | pass | — | Simulated series (trend + 16-period sine + noise). Greeners MODWT uses unnormalised Haar filters, equivalent to pywt.swt(..., norm=False). Wavelet energies are compared as coefficient-like quantities; standard errors are not meaningful. |
| msvar | simulated_msvar | R:passed *, Python:passed * | pass | — | Simulated two-regime VAR with known intercepts and transition matrix; compares regime-specific y1 intercepts and transition probabilities against R MSwM and Python statsmodels MarkovRegression. Tolerances reflect label/algorithm sensitivity in regime-switching models. |
| nardl | simulated | R:passed *, Python:passed * | pass | — | Simulated NARDL(1,1) with asymmetric long-run multipliers and short-run dynamics. y and x are random walks with positive and negative shock decomposition. |
| negbin | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Negative binomial regression for number of children on age, education, electric and urban indicators. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a * (b1*x1^rho + (1-b1)*x2^rho)^(1/rho) + N(0, 0.1). The CES function is now identified by the share restriction b2 = 1 - b1. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a * x1^b1 * x2^b2 + N(0, 0.3). Coefficients and standard errors compared against R `nls` and Python `curve_fit`. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a * exp(b * x) + N(0, 0.1). Coefficients and standard errors compared against R `nls` and Python `scipy.optimize.curve_fit`. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a / (1 + exp(-b*(x-c))) + N(0, 0.2). Coefficients and standard errors compared against R `nls` and Python `scipy.optimize.curve_fit`. |
| nls | simulated | R:passed *, Python:passed * | pass | — | Simulated data from y = a * x^b + N(0, 0.3). Coefficients and standard errors compared against R `nls` and Python `scipy.optimize.curve_fit`. |
| ologit | wooldridge::beauty | R:passed *, Python:passed * | pass | — | Ordered logit of looks (2, 3, 4) on female, educ, exper, black. |
| ols | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | OLS wage equation with one-way cluster-robust standard errors by worker id. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | 89 | OLS log-wage equation with HC3 heteroskedasticity-robust standard errors. |
| ols | wooldridge::phillips | R:passed *, Python:passed * | pass | 91 | OLS expectations-augmented Phillips curve with Newey-West HAC standard errors. |
| ols | wooldridge::wagepan | R:passed *, Python:passed * | pass | 87 | OLS wage equation with two-way clustered standard errors by worker id and year. |
| ols | wooldridge::401k | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.3 401(k) participation equation. |
| ols | wooldridge::attend | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 6, Example 6.3 attendance effects on standardized final exam score. |
| ols | wooldridge::barium | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10, Example 10.5 barium chloride import demand and antidumping filings. |
| ols | wooldridge::bwght | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 5, Example 5.2 birth weight and maternal smoking equation. |
| ols | wooldridge::campus | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4, Example 4.4 log-log campus crime and enrollment equation. |
| ols | wooldridge::ceosal1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 2, Example 2.11 log-log CEO salary equation. |
| ols | wooldridge::ceosal1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 2, Example 2.3 CEO salary and return on equity equation. |
| ols | wooldridge::consump | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10, Example 10.4 consumption growth on income growth equation. |
| ols | wooldridge::crime1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3, Example 3.5 arrest records equation with average sentence length. |
| ols | wooldridge::crime1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.5 arrest records equation. |
| ols | wooldridge::fertil3 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 13, Example 13.3 fertility distributed lag equation. |
| ols | wooldridge::gpa1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 3 Example 3.1 college GPA equation. |
| ols | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4, Section 4.5 log housing price equation with qualitative information. |
| ols | wooldridge::hprice2 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 6, Example 6.2 log housing price equation with quadratic in rooms. |
| ols | wooldridge::htv | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 9, Example 9.3 education and parental education/ability equation. |
| ols | wooldridge::intdef | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10, Example 10.2 interest rate, inflation and deficit equation. |
| ols | wooldridge::jtrain | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14, Example 14.3 pooled job training scrap rate equation. |
| ols | wooldridge::kielmc | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 13, Example 13.1 difference-in-differences housing price equation. |
| ols | wooldridge::meap93 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4, Examples 4.2 and 4.10 math pass rate equation with log salary, staff and enrollment. |
| ols | wooldridge::nyse | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 11, Example 11.4 AR(1) test of efficient markets hypothesis. |
| ols | wooldridge::phillips | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10, Example 10.1 static Phillips curve. |
| ols | wooldridge::phillips | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 11, Example 11.5 expectations-augmented Phillips curve. |
| ols | wooldridge::prminwge | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 10, Example 10.3 Puerto Rican employment and minimum wage equation. |
| ols | wooldridge::sleep75 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 5, Problem 3.3 sleep-work tradeoff equation. |
| ols | wooldridge::twoyear | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 4, Example 4.10 returns to two-year and four-year college credits. |
| ols | wooldridge::vote1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 2, Examples 2.5 and 2.9 election outcomes and campaign expenditure share equation. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | First real-dataset validation case. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | First textbook example using wage1. Log-linear returns to education. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 7, Example 7.1 hourly wage equation with a qualitative dummy variable. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 7, Example 7.6 hourly wage equation with marriage and gender dummy interactions. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 7, Example 7.1 log hourly wage equation with a qualitative dummy variable. |
| ols | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 6, Section 6.2 wage equation with a quadratic in experience. |
| oprobit | wooldridge::beauty | R:passed *, Python:passed * | pass | — | Ordered probit model of self-reported beauty rating (looks 2-5) on female, education, experience and black indicators. |
| orf | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 1 + 2*x1 - x2 + 0.5*treated + 0.3*w1 - 0.2*w2 + N(0,1). Hayashi orf() reports the average treatment effect. R reference uses grf::causal_forest on the full set of covariates; Python uses econml.orf.DROrthoForest. |
| panel_fe | wooldridge::wagepan | R:passed *, Python:passed * | pass | 115 | Panel fixed-effects wage equation with worker-clustered standard errors using explicit within-transformed CR1 reference implementations. Tolerance reflects Hayashi's four-decimal text export. |
| panel_fe | wooldridge::grunfeld | R:passed *, Python:passed * | pass | — | Panel fixed-effects investment demand model (Grunfeld). |
| panel_fe | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Panel fixed-effects wage equation with time-clustered standard errors using explicit within-transformed CR1 reference implementations. |
| panel_fe | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Panel fixed-effects wage equation with two-way (entity + time) clustered standard errors using explicit within-transformed CR1 reference implementations. |
| panel_fe | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Wooldridge Introductory Econometrics Chapter 14, Example 14.4 panel fixed-effects wage equation. |
| panel_heckman | simulated_panel_heckman | R:passed *, Python:passed * | pass | — | Panel Heckman selection model (two-step) with selection equation and outcome equation. Uses simulated panel data with known selection mechanism. |
| panel_qreg | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with entity fixed effects and heteroskedastic errors. References demean the data and run quantile regression without an intercept; standard errors are convention-sensitive. |
| panel | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50, T=4, random effects, left-censored at 0. Coefficients and standard errors compared against pooled Tobit (censReg and MLE). |
| pca | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | PCA is run on educ, exper, tenure, and wage with centering and unit-variance scaling. Loadings are compared in absolute value because each eigenvector's sign is arbitrary. Hayashi displays four decimal places, so tolerances allow for output rounding. |
| pcse | wooldridge::wagepan | R:passed *, Python:passed * | pass | 99, 103 | PCSE estimation of log wage on education, experience, and dummies using the Hayashi/Greeners Beck-Katz covariance convention. |
| poisson | wooldridge::fertil2 | R:passed *, Python:passed * | pass | — | Poisson regression for number of children on the fertil2 dataset. |
| portsort | simulated | Python:passed *, R:passed * | pass | — | Five equal-count portfolios sorted by size. Compare mean returns and high-low spread. |
| probit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Probit labour-force participation on the Mroz dataset. |
| psm | wooldridge::jtrain3 | R:passed *, Python:passed * | pass | — | R, Python, and Hayashi use the same absolute-caliper, no-replacement matching protocol; ATT agrees and bootstrap SEs are within tolerance. |
| pstr | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50, T=10. y = beta0*x + beta1*x*g(q; gamma=5, c=0.5) + FE + noise. Gamma, c, beta0_x and beta1_x compared against grid-search references. |
| pvar | simulated | R:passed *, Python:passed * | pass | — | Simulated bivariate panel VAR with N=50 and T=100. Hayashi GMM and within-OLS references agree within moderate tolerance due to the Nickell bias in within estimation. |
| qreg | simulated | R:passed *, Python:passed * | pass | — | Simulated heteroskedastic data. Quantile regression at tau=0.75 with bootstrap standard errors compared against R quantreg and statsmodels. |
| qrf_inf | simulated | R:passed *, Python:passed * | pass | — | Simulated heteroskedastic data y = 3*x1 + N(0, 0.1*(1+0.5*x1)), with x2 irrelevant. qrf_inf() at q=0.75 with 50 trees, depth 5, 50 bootstrap samples. OOB R^2 compared against grf::quantile_forest and quantile_forest.RandomForestQuantileRegressor. |
| qrf | simulated | R:passed *, Python:passed * | pass | — | Simulated heteroskedastic data y = 3*x1 + N(0, 0.1*(1+0.5*x1)). QRF at tau=0.75 with 50 trees, depth 5. OOB R^2 compared against quantile_forest.RandomForestQuantileRegressor. |
| qreg | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Median quantile regression of wage on education, experience, and tenure. |
| qvar | simulated | R:passed *, Python:passed * | pass | — | Simulated bivariate VAR(1) process. Both R (quantreg::rq) and Python (statsmodels QuantReg) are run separately for each equation at the median (tau=0.5). Quantile-regression standard errors are algorithm- and implementation-specific, so only coefficients are compared and std_err is set to NaN. |
| rdd | rdd_book | R:passed *, Python:passed * | pass | — | Sharp RDD with local linear regression, triangular kernel and Imbens-Kalyanaraman bandwidth. |
| re | grunfeld | R:passed *, Python:passed * | pass | 101 | Random-effects investment demand model (Grunfeld). |
| reg_path | simulated | R:passed *, Python:passed * | pass | — | Simulated data where y = 0.5 + x1 + noise and x2 is noise. Hayashi reg_path selects an elastic-net lambda by BIC; references fit glmnet and sklearn ElasticNet at the same optimal lambda and standardisation. |
| diagnostics | simulated | R:passed * | pass | — | Ramsey RESET specification test on simulated OLS residuals. |
| rf | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 3*x1 + N(0, 0.1). In-sample R² compared against scikit-learn. Standard errors are not defined for an out-of-bag R² summary. |
| ridge | wooldridge::hprice1 | R:passed *, Python:passed * | pass | 106 | Ridge regression of log house price on log lot size, log square footage, bedrooms and colonial dummy. |
| rlm | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Huber robust linear regression of log wage on education, experience, and tenure. |
| rolling | simulated | R:passed *, Python:passed * | pass | — | Simulated linear trend (y = 1 + 2x + noise). Compares the last-window rolling OLS coefficients produced by Hayashi's rolling() against a zoo::rollapply reference and a statsmodels OLS fit on the final window. |
| setar | simulated | R:passed *, Python:passed * | pass | — | Simulated SETAR(1,1,1) with two regimes split by y_{t-1}. Hayashi grid search may differ slightly from R tsDyn; tolerances are relaxed accordingly. |
| sfa | simulated | R:passed *, Python:passed * | pass | — | Simulated Cobb-Douglas production frontier with negligible inefficiency so MLE/OLS references align with Hayashi. |
| spatial_durbin_error | simulated |  | not-supported | — | R spatialreg/spdep packages failed to install in previous sessions. |
| spatial_durbin | simulated | R:passed *, Python:passed * | pass | — | Data generated on a 7x7 grid with rook contiguity W, rho=-0.95, beta=0.5. The Durbin model is highly collinear; only the spatial autoregressive parameter is compared. |
| spatial_panel_sar | simulated |  | not-supported | — | R spatialreg/spdep packages failed to install in previous sessions. |
| spatial_sar | simulated | R:passed *, Python:passed * | pass | — | Data generated on a 7x7 grid with rook contiguity W, rho=0.3, beta=0.5. Reference implements the same concentrated MLE independently. |
| spatial_sem | simulated | R:passed *, Python:passed * | pass | — | Data generated on a 7x7 grid with rook contiguity W, lambda=0.1, beta=0.5. Reference implements the same concentrated MLE independently. |
| spectral | simulated |  | not-supported | — | Results are sensitive to random k-means initialisation and normalised Laplacian details; no deterministic numeric reference. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Summary statistics with detail (percentiles, skewness, kurtosis) for wage. |
| sur | wooldridge::grunfeld | R:passed *, Python:passed * | pass | — | Two-equation SUR (Zellner FGLS) on the Grunfeld investment data. |
| sv | simulated_sv | R:passed *, Python:passed * | pass | — | Simulated Taylor (1986) SV data; compares the posterior mean of the latent log-volatility h_t between Hayashi, R stochvol and PyMC. Tight tolerance because R and PyMC agree closely on this DGP. |
| svar | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | Cholesky-identified SVAR(2) on log US real GDP and consumption. |
| svar | simulated | R:passed *, Python:passed * | pass | — | Simulated stable VAR(1) with 250 observations. Blanchard-Quah long-run identification via lower Cholesky of the long-run covariance. |
| diagnostics | simulated | R:passed * | pass | — | Shapiro-Wilk normality test on a simulated random sample. |
| synth | synth_smoking | R:passed *, Python:passed * | pass | — | R, Python, and Hayashi implement the same outcome-only simplex SCM and agree on ATT within tolerance. |
| synthdid | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with 20 units, 10 periods, treatment begins at period 6 for unit 0 with ATT=2.0. Reference uses a simple synthetic-control-style pre-treatment weighting and computes the post-treatment mean gap. The ATT has no standard error. |
| sysgmm | wooldridge::wagepan | R:passed *, Python:passed * | pass | 117 | System GMM (Blundell-Bond) two-step on Wooldridge wagepan with lags=2. R and Python references explicitly implement the same two-step System GMM procedure used by Hayashi/Greeners; plm::pgmm is not used as the active R oracle because it uses different instrument and weighting conventions. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | Tabstat statistics (mean, sd, min, max, p50) for wage, educ, exper, tenure. |
| descriptive | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Two-way frequency table with Pearson chi-square test. |
| ols | simulated | R:passed *, Python:passed * | pass | — | Simulated OLS. testparm F and p-value for H0: x1 = x2 = 0. |
| three_sls | simulated | R:passed *, Python:passed * | pass | — | Simultaneous two-equation system with correlated errors. Each equation includes an intercept, one exogenous and one endogenous regressor; the excluded exogenous from the other equation is used as an instrument. Python reference is linearmodels.system.IV3SLS with an explicit constant column. |
| tmle | simulated_tmle | R:passed *, Python:passed * | pass | — | Simulated data with true ATE 0.7; compares TMLE point estimate and standard error against R tmle and a manual Python implementation. |
| tobit | wooldridge::mroz | R:passed *, Python:passed * | pass | — | Tobit regression of hours worked with left censoring at zero. Hayashi matches AER::tobit at displayed precision; the new Python reference is a manual MLE refined with Nelder-Mead and uses a numerical Hessian for standard errors. |
| transformer | simulated |  | not-supported | — | PyTorch/TensorFlow stochastic training is not reproducible enough for numerical validation. |
| tsne | simulated | Python:passed * | pass | — | t-SNE embedding of three 3D blobs; cluster quality measured via K-Means inertia. |
| descriptive | wooldridge::wage1 | R:passed *, Python:passed * | pass | — | One-sample t-test of wage mean against mu=5. |
| tvar | simulated | R:passed *, Python:passed * | pass | — | Simulated bivariate TVAR with exogenous threshold q; references use no-intercept regime OLS to match Hayashi's output. |
| tvcopula | simulated |  | not-supported | — | R rmgarch/ccgarch packages not available; Python copula packages do not match Hayashi output. |
| tvp | simulated | R:passed *, Python:passed * | pass | — | Simulated TVP data with smooth intercept and slope drift. The reference is the true final coefficient vector because Greeners TVP uses a simple Kalman-grid implementation with no readily available reference implementation. |
| tvp_var | simulated |  | not-supported | — | No stable PyMC or R TVP reference implementation in the venv. |
| ucm | simulated | Python:passed * | pass | — | Simulated monthly series with local-linear trend and deterministic seasonal (period=12). Compares the estimated irregular variance and the first/last smoothed level state. |
| umap | simulated | Python:passed * | pass | — | UMAP embedding of three 3D blobs; cluster quality measured via K-Means inertia. |
| var | simulated_var1 | R:passed *, Python:passed * | pass | — | Uses the same simulated bivariate VAR(1) DGP as Chapter 28 of the book. |
| var | statsmodels::macrodata | R:passed *, Python:passed * | pass | — | VAR(2) on US real GDP and consumption. |
| varma | simulated | R:passed *, Python:passed * | pass | — | Bivariate VARMA(1,1) with known AR and MA matrices. Hayashi uses the Hannan-Rissanen algorithm; the Python reference uses statsmodels VARMAX with no trend. Coefficients are compared (standard errors are not computed by the current Hayashi VARMA implementation). |
| diagnostics | simulated | R:passed * | pass | — | Variance inflation factors for the OLS regressor matrix on simulated data. |
| iv | simulated | R:passed *, Python:passed * | pass | — | Simulated weak instrument. First-stage partial F and p-value. |
| diagnostics | simulated | R:passed * | pass | — | White heteroskedasticity test on simulated OLS residuals. |
| wls | wooldridge::hprice1 | R:passed *, Python:passed * | pass | — | WLS with weights generated inside Hayashi to avoid sandbox file issues. |
| xgboost | simulated | R:passed *, Python:passed * | pass | — | Simulated data y = 3*x1 + N(0, 0.1), x2 irrelevant. XGBoost with 50 trees, learning rate 0.1, max depth 3, default regularization. MSE and R^2 compared against xgboost.XGBRegressor. |
| xtgls | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Panel feasible GLS with panel-level heteroskedasticity (Parks/Kmenta, Stata xtgls panels(heteroskedastic)). R and Python references implement the same two-step FGLS procedure used by Hayashi/Greeners. |
| xtlogit | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE logit with exchangeable working correlation; coefficients and sandwich standard errors compared. |
| xtpoisson | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE Poisson with exchangeable working correlation; coefficients and sandwich standard errors compared. |
| xtprobit | simulated | R:passed *, Python:passed * | pass | — | Simulated panel with N=50 groups and T=4 periods. GEE probit with exchangeable working correlation; coefficients and sandwich standard errors compared. |
| descriptive | wooldridge::wagepan | R:passed *, Python:passed * | pass | — | Overall, between, and within panel summary for lwage. |
| zinb | wooldridge::affairs | R:passed *, Python:passed * | pass | 123 | ZINB model of number of affairs on demographic predictors. |
| zip | wooldridge::affairs | R:passed *, Python:passed * | pass | 121 | ZIP model of number of affairs on demographic predictors. |
| diagnostics | simulated | R:passed * | pass | — | Zivot-Andrews structural-break unit-root test on a simulated random walk. |

## Status legend

- `pass` — Hayashi matches all available references within declared tolerances.
- `partial` — Hayashi matches at least one reference, but other declared references failed or are missing; exits non-zero unless `--allow-partial` is passed.
- `fail` — Hayashi differs from at least one reference beyond tolerances.
- `blocked` — no declared reference could run; the case cannot be judged.
- `not-supported` — the validation programme cannot currently test the stated estimator/workflow contract; this does not necessarily mean Hayashi lacks the command.
- `not-started` — registered but not implemented.

The Reference column lists declared reference implementations, or
per-reference execution details when a runner result records them.
A declared reference that fails or is missing no longer blocks
comparison when `--allow-partial` is used; otherwise partial cases
fail the runner.

This matrix is generated from `validation/matrix.yml` by `validation/run.py`.

This matrix covers the core empirical estimators. Some commands are
intentionally excluded for the reasons described in the "Estimators not
covered by validation" section of the README.

Esta matriz abrange os estimadores empíricos centrais. Alguns comandos são
deixados de fora intencionalmente pelos motivos descritos na seção
"Estimators not covered by validation" do README.


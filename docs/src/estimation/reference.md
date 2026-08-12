# Estimator Reference

Quick reference for the implemented estimator and model commands in Hayashi.
Aliases are shown with `/`. Common post-estimation commands are listed at the
end of the page.

## Cross-Section

| Command | Description | Syntax |
|---------|-------------|--------|
| `ols` / `reg` | OLS linear regression | `ols(Y ~ X1 + X2, df)` |
| `wls` | Weighted least squares | `wls(Y ~ X1 + X2, df, weights="w")` |
| `iv` | Instrumental variables / 2SLS | `iv(Y ~ X_exog + X_endo, ~ Z + X_exog, df)` |
| `logit` | Logistic regression | `logit(Y ~ X1 + X2, df)` |
| `probit` | Probit regression | `probit(Y ~ X1 + X2, df)` |
| `ologit` | Ordered logit | `ologit(Y ~ X1 + X2, df)` |
| `oprobit` | Ordered probit | `oprobit(Y ~ X1 + X2, df)` |
| `mlogit` | Multinomial logit | `mlogit(Y ~ X1 + X2, df, base=1)` |
| `cmnlogit` | Conditional multinomial logit | `cmnlogit(choice ~ price + quality, df, group=id, alts=3)` |
| `clogit` | Conditional logit | `clogit(Y ~ X1 + X2, df, group=id)` |
| `cpoisson` | Conditional Poisson / PPML | `cpoisson(Y ~ X1 + X2, df, group=id)` |
| `tobit` | Tobit (censored regression) | `tobit(Y ~ X1 + X2, df, ll=0)` |
| `heckman` / `heckit` | Heckman selection model | `heckman(Y ~ X1, S ~ Z1 + Z2, df)` |
| `qreg` | Quantile regression | `qreg(Y ~ X1 + X2, df, q=0.5)` |
| `nbreg` | Negative binomial regression | `nbreg(Y ~ X1 + X2, df)` |
| `poisson` | Poisson regression | `poisson(Y ~ X1 + X2, df)` |
| `zip` / `zinb` | Zero-inflated count models | `zip(Y ~ X1, df, inflate=["Z1", "Z2"])` |
| `rlm` | Robust M-estimation | `rlm(Y ~ X1 + X2, df)` |
| `glm` | Generalized linear model | `glm(Y ~ X1 + X2, df, family=poisson)` |
| `gee` | Generalized estimating equations | `gee(Y ~ X1 + X2, df, id=group)` |
| `betareg` | Beta regression | `betareg(share ~ X1 + X2, df)` |

## Panel Data

| Command | Description | Syntax |
|---------|-------------|--------|
| `fe` | Fixed effects (within estimator) | `fe(Y ~ X1 + X2, df)` |
| `re` | Random effects (GLS) | `re(Y ~ X1 + X2, df)` |
| `feiv` | FE with instrumental variables | `feiv(Y ~ X_exog + X_endo, ~ Z, df)` |
| `ab` | Arellano-Bond | `ab(Y ~ X1 + X2, df, id=firm, time=year)` |
| `sysgmm` | System GMM | `sysgmm(Y ~ X1 + X2, df, id=firm, time=year)` |
| `pcse` | Panel-corrected standard errors | `pcse(Y ~ X1 + X2, df, id=firm, time=year)` |
| `xtgls` | Feasible GLS for panels | `xtgls(Y ~ X1 + X2, df, id=firm, time=year)` |
| `pthresh` | Panel threshold model | `pthresh(Y ~ X1, df, id=firm, q=threshold_var)` |

## Time Series

| Command | Description | Syntax |
|---------|-------------|--------|
| `arima` / `sarima` | ARIMA / SARIMA | `arima(df, Y, p=1, d=1, q=1)` |
| `autoreg` | Autoregression | `autoreg(df, Y, lags=2)` |
| `ardl` | Autoregressive distributed lag model | `ardl(df, Y, X, p=2, q=1)` |
| `kalman` | State-space Kalman smoothing | `kalman(df, Y, model="ll")` |
| `garch` / `egarch` / `gjrgarch` | Volatility models | `garch(df, Y, p=1, q=1)` |
| `var` | Vector autoregression | `var(df, Y1, Y2, lags=2)` |
| `vecm` | Vector error correction | `vecm(df, Y1, Y2, lags=2, rank=1)` |
| `varma` | VARMA / VARMAX | `varma(df, [Y1, Y2], p=1, q=1)` |
| `svar` | Structural VAR | `svar(df, Y1, Y2, lags=2, type=short)` |
| `ucm` | Unobserved components model | `ucm(df, Y)` |
| `ets` | Exponential smoothing | `ets(df, Y)` |
| `msauto` | Markov-switching autoregression | `msauto(df, Y, regimes=2)` |
| `decompose` / `stl` / `mstl` | Series decomposition | `stl(df, Y, period=12)` |

## Causal Inference

| Command | Description | Syntax |
|---------|-------------|--------|
| `did` | Difference-in-differences | `did(Y ~ X, df, treat=D, post=P)` |
| `rd` | Sharp regression discontinuity | `rd(Y ~ running, cutoff, df)` |
| `fuzzy_rd` | Fuzzy regression discontinuity | `fuzzy_rd(Y ~ running, "treatment", cutoff, df)` |
| `synth` | Synthetic control | `synth("Y", "treated_id", t0, df, id="unit", time="year")` |
| `psm` | Propensity score matching | `psm(Y ~ treated + X1 + X2, df)` |

## Finance

| Command | Description | Syntax |
|---------|-------------|--------|
| `fmb` | Fama-MacBeth regression | `fmb(ret ~ beta + size + bm, df, time=month)` |
| `portsort` | Portfolio sort | `portsort(df, ret, size, n=5)` |
| `doublesort` | Two-way portfolio sort | `doublesort(df, ret, size, bm, n1=5, n2=5)` |

## Multivariate and Dimension Reduction

| Command | Description | Syntax |
|---------|-------------|--------|
| `sur` / `sureg` | Seemingly unrelated regressions | `sur(df, Y1 ~ X1, Y2 ~ X2)` |
| `three_sls` / `threesl` | Three-stage least squares | `threesl(df, Y1 ~ X1, Y2 ~ X2, instruments=["Z1"])` |
| `pca` | Principal component analysis | `pca(df, [X1, X2, X3])` |
| `factor` | Factor analysis | `factor(df, [X1, X2, X3])` |
| `dfm` | Dynamic factor model | `dfm(df, Y1, Y2, factors=2)` |
| `manova` | Multivariate ANOVA | `manova(df, [Y1, Y2], by=group)` |
| `cancorr` | Canonical correlation | `cancorr(df, [X1, X2], [Y1, Y2])` |

## Smoothing, Imputation, and Flexible Models

| Command | Description | Syntax |
|---------|-------------|--------|
| `lowess` | Local polynomial smoothing | `lowess(df, Y, X, frac=0.3)` |
| `gam` | Generalized additive model | `gam(Y ~ X1 + X2, df)` |
| `mice` | Multiple imputation by chained equations | `mice(df, vars=["Y", "X1", "X2"])` |

## Regularization

| Command | Description | Syntax |
|---------|-------------|--------|
| `lasso` | LASSO (L1 penalty) | `lasso(Y ~ X1 + X2 + ... + Xp, df)` |
| `ridge` | Ridge (L2 penalty) | `ridge(Y ~ X1 + X2 + ... + Xp, df)` |
| `elasticnet` | Elastic net | `elasticnet(Y ~ X1 + ... + Xp, df, alpha=0.5)` |

## Survival Analysis

| Command | Description | Syntax |
|---------|-------------|--------|
| `cox` | Cox proportional hazards | `cox(T ~ X1 + X2, df, event=D)` |
| `km` | Kaplan-Meier survival curve | `km(df, time=t, event=d)` |

## Common Options

Common options vary by command. Check `help(command)` in the REPL for the
supported options of a specific estimator.

| Option | Description |
|--------|-------------|
| `cov=robust` | Heteroskedasticity-robust SE (HC1) |
| `cov=hc0` ... `cov=hc4` | Specific HC variant |
| `cluster=var` | Cluster-robust SE |
| `cluster=var, cluster2=var2` | Two-way cluster SE |
| `nw=L` | Newey-West HAC SE with L lags |
| `if=(condition)` | Subsample estimation |
| `bootstrap(est, formula, df, n=N)` | Bootstrap standard errors |

## Machine Learning and Causal ML

| Command | Description | Syntax |
|---------|-------------|--------|
| `rf` / `random_forest` | Random Forest regression (Breiman 2001) | `rf(Y ~ X1 + X2, df)` |
| `gbm` / `gradient_boosting` | Gradient Boosting regression (Friedman 2001) | `gbm(Y ~ X1 + X2, df)` |
| `qrf` / `quantile_forest` | Quantile Regression Forest (Meinshausen 2006) | `qrf(Y ~ X1 + X2, df, quantiles="0.1,0.5,0.9")` |
| `xgboost` / `xgb` | XGBoost (Chen & Guestrin 2016) | `xgboost(Y ~ X1 + X2, df)` |
| `mlp` / `neural_net` | Multilayer Perceptron | `mlp(Y ~ X1 + X2, df)` |
| `lstm` | Long short-term memory network | `lstm(df, Y, hidden=10, seqlen=12, forecast=5)` |
| `transformer` / `transformer_ts` | Transformer for time series | `transformer(df, Y, d_model=8, seqlen=12, forecast=5)` |
| `bart` / `bayesian_trees` | Bayesian Additive Regression Trees | `bart(Y ~ X1 + X2, df)` |
| `gp` / `gaussian_process` | Gaussian Process regression | `gp(Y ~ X1 + X2, df)` |
| `causalforest` / `causal_forest` | Causal Forest (Wager-Athey 2018) | `causalforest(Y ~ treated, df, x="age,income")` |
| `grf` / `generalized_rf` | Generalized Random Forest (Athey-Tibshirani-Wager 2019) | `grf(Y ~ treated, df, x="age,income")` |
| `dr_learner` / `drlearner` | DR-Learner (Kennedy 2023) | `dr_learner(Y ~ treated, df, x="X1,X2")` |
| `tmle` | Targeted Maximum Likelihood Estimation | `tmle(Y ~ treated, df, w="X1,X2")` |
| `orf` / `orthogonal_forest` | Orthogonal Random Forest (Oprescu-Syrgkanis-Wu 2019) | `orf(Y ~ treated, df, x="X1,X2")` |

## Clustering and Unsupervised Learning

| Command | Description | Syntax |
|---------|-------------|--------|
| `kmeans` / `k_means` | K-means clustering | `kmeans(df, x="X1,X2", k=3)` |
| `dbscan` / `dbscan_clust` | DBSCAN density-based clustering | `dbscan(df, x="X1,X2", eps=0.5, min=5)` |
| `hclust` / `hierarchical` | Hierarchical clustering | `hclust(df, x="X1,X2", k=3)` |
| `tsne` / `t_sne` | t-SNE dimensionality reduction | `tsne(df, x="X1,X2,X3", k=2)` |
| `umap` | UMAP dimensionality reduction | `umap(df, x="X1,X2,X3", k=2)` |
| `spectral` / `spectral_clustering` | Spectral clustering | `spectal(df, x="X1,X2", k=3)` |
| `gmm_clust` / `gmm_clustering` | Gaussian Mixture Model clustering | `gmm_clust(df, x="X1,X2", k=3)` |
| `biplot` / `pca_biplot` | PCA biplot | `biplot(df, x="X1,X2,X3")` |
| `kde` | Kernel density estimation | `kde(df, X, bw=0.5, kernel="gaussian")` |

## Spatial Econometrics

| Command | Description | Syntax |
|---------|-------------|--------|
| `spatial_sar` | Spatial autoregressive model | `spatial_sar(Y ~ X1 + X2, df, w=W)` |
| `spatial_sem` | Spatial error model | `spatial_sem(Y ~ X1 + X2, df, w=W)` |
| `spatial_durbin` / `sdm` | Spatial Durbin model | `spatial_durbin(Y ~ X1 + X2, df, w=W, id="entity")` |
| `spatial_panel_sar` | Spatial panel SAR | `spatial_panel_sar(Y ~ X1 + X2, df, w=W, id="entity")` |
| `spatial_panel_sem` | Spatial panel SEM | `spatial_panel_sem(Y ~ X1 + X2, df, w=W, id="entity")` |
| `spatial_durbin_error` / `sdem` | Spatial Durbin error model | `spatial_durbin_error(Y ~ X1 + X2, df, w=W, id="entity")` |

## Advanced Time Series and Volatility

| Command | Description | Syntax |
|---------|-------------|--------|
| `msauto` / `markov_ar` | Markov-switching autoregression | `msauto(df, Y, regimes=2, lags=1)` |
| `varma` / `varmax` | VARMA / VARMAX | `varma(df, [Y1, Y2], p=1, q=1)` |
| `svar` | Structural VAR (Cholesky) | `svar(df, [Y1, Y2], lags=2)` |
| `sirf` / `svar_irf` | SVAR impulse response functions | `sirf(v, n=10)` |
| `sfevd` / `svar_fevd` | SVAR forecast error variance decomposition | `sfevd(v, n=10)` |
| `threesl` / `three_sls` / `3sls` | Three-stage least squares | `threesl(df, Y1 ~ X1, Y2 ~ X2)` |
| `modwt` | Maximal overlap discrete wavelet transform | `modwt(df, Y, scales=4)` |
| `copula` | Copula-based dependence modelling | `copula(Y1 ~ Y2, df, type="gaussian")` |
| `tvcopula` / `tv_copula` | Time-varying copula | `tvcopula(Y1 ~ Y2, df, type="gaussian")` |
| `sv` / `stochastic_vol` | Stochastic volatility (HRS QMLE) | `sv(df, Y)` |
| `dcc_garch` / `dcc` | Dynamic Conditional Correlation GARCH | `dcc_garch(Y1 ~ Y2, df)` |
| `tvar` / `threshold_var` | Threshold VAR | `tvar(Y1 ~ Y2, df, q="threshold_var", lags=1)` |
| `bvar` / `bayesian_var` | Bayesian VAR with Minnesota prior | `bvar(Y1 ~ Y2, df, lags=1)` |
| `mfvar` / `mixed_freq_var` | Mixed-frequency VAR | `mfvar(df_low, Y_low, df_high, Y_high)` |
| `qvar` / `quantile_var` | Quantile VAR | `qvar(Y1 ~ Y2, df, lags=1, tau=0.5)` |
| `setar` | Self-exciting threshold AR | `setar(Y, df, order=2, delay=1)` |
| `nardl` | Nonlinear ARDL | `nardl(Y ~ X, df, lags=1)` |
| `midas` | Mixed Data Sampling regression | `midas(Y ~ X, df, freq=3, lags=12, poly=2)` |
| `tvp` | Time-varying parameter regression | `tvp(Y ~ X1 + X2, df)` |
| `tvp_var` | Time-varying parameter VAR | `tvp_var(Y1 ~ Y2, df, lags=1)` |
| `hawkes` | Hawkes self-exciting point process | `hawkes(df, time_var, T=100)` |

## Bayesian and Frontier Models

| Command | Description | Syntax |
|---------|-------------|--------|
| `bayes_lm` / `bayesian_lm` | Bayesian linear regression | `bayes_lm(Y ~ X1 + X2, df)` |
| `bayes_sfa_production` / `bayes_frontier` | Bayesian SFA production | `bayes_sfa_production(Y ~ X1 + X2, df)` |
| `bayes_sfa_cost` | Bayesian SFA cost | `bayes_sfa_cost(Y ~ X1 + X2, df)` |
| `sfa_production` / `frontier` | Stochastic frontier analysis (production) | `sfa_production(Y ~ X1 + X2, df)` |
| `sfa_cost` | Stochastic frontier analysis (cost) | `sfa_cost(Y ~ X1 + X2, df)` |

## Panel and Advanced Models

| Command | Description | Syntax |
|---------|-------------|--------|
| `panel_tobit` | Panel Tobit with random effects | `panel_tobit(Y ~ X1 + X2, df, id="firm")` |
| `panel_heckman` | Panel Heckman with random effects | `panel_heckman(Y ~ X1 + X2, df, sel="Z ~ W", id="firm")` |
| `panel_qreg` / `panel_quantile` | Panel quantile regression | `panel_qreg(Y ~ X1 + X2, df, id="firm")` |
| `pvar` / `panel_var` | Panel VAR (GMM) | `pvar(Y1 ~ Y2, df, id="entity", lags=1)` |
| `msvar` / `ms_var` | Markov-switching VAR | `msvar(Y1 + Y2, df, regimes=2, lags=1)` |
| `favar` | Factor-augmented VAR | `favar(Y1 ~ Y2 + Y3, df, observed="rate", factors=2)` |
| `fapanel` / `fa_panel` | Factor-augmented panel | `fapanel(Y ~ X1 + X2, df, aux="aux_df", id="entity")` |
| `pstr` | Panel Smooth Transition Regression | `pstr(Y ~ X1 + X2, df, q="transition_var", id="entity")` |
| `fcoef` / `functional_coef` | Functional coefficient model | `fcoef(Y ~ X1 + X2, df, z="moderator")` |
| `fmols` | Fully Modified OLS | `fmols(Y ~ X1 + X2, df)` |

## Tests and Diagnostics

| Command | Description |
|---------|-------------|
| `test(m, ...)` | Joint or restriction tests |
| `testparm(m, vars)` | Joint F-test for selected variables |
| `estat_overid` / `sargan` | Overidentification test |
| `estat_endog` / `dwh` | Durbin-Wu-Hausman endogeneity test |
| `estat_classification` | Classification table for logit/probit |
| `estat_gof` / `hltest` | Hosmer-Lemeshow GOF test |
| `linktest(m)` | Specification error link test |
| `lroc(m)` | ROC / AUC / Gini |
| `reset(m)` | Ramsey RESET test |
| `white(m)` | White heteroskedasticity test |
| `durbinwatson` / `dw` | Durbin-Watson statistic |
| `jb` / `bgodfrey` / `archtest` | Normality / serial correlation / ARCH tests |
| `adf` / `kpss` / `pp` / `za` | Unit-root tests |
| `granger` / `engle_granger` / `johansen` | Causality / cointegration tests |
| `swilk` / `sfrancia` / `sktest` | Normality tests |
| `adtest` / `lilliefors` / `harveycollier` | Additional normality / misspecification tests |
| `spearman` / `ranksum` / `kruskal` / `signrank` | Non-parametric tests |
| `proptest` / `proptest2` / `propci` | Proportion tests |
| `chisq2x2` | 2x2 chi-square test |
| `multipletests` | Multiple-testing correction |

## Post-Estimation

| Command | Description |
|---------|-------------|
| `tidy(m)` | Tidy coefficient table (variable, coef, std_err, t/z, p_value, conf_low, conf_high) |
| `glance(m)` | Model fit statistics (r2, adj_r2, AIC, BIC, log_lik, n, sigma, etc.) |
| `esttab(m1, m2, ...)` | Side-by-side estimation table |
| `predict df var = m [, "kind"]` | Fitted values, residuals, probabilities |
| `margins(m, type=ame)` | Average marginal effects |
| `hausman(m_fe, m_re)` | Hausman specification test |
| `hausman_robust(m_fe, m_re)` | Robust Hausman test (Wooldridge 2010) |
| `ftest_robust(m [, vars=])` | Robust F-test (Wooldridge 2010) |
| `irf(v, ...)` | Impulse response functions |
| `johansen(...)` | Johansen cointegration test |
| `testparm(m, vars)` | Joint significance test |
| `vif(m)` | Variance inflation factors |

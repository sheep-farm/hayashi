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
| `be` | Between estimator | `be(Y ~ X1 + X2, df)` |
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

## Post-Estimation

| Command | Description |
|---------|-------------|
| `esttab(m1, m2, ...)` | Side-by-side estimation table |
| `predict df var = m [, "kind"]` | Fitted values, residuals, probabilities |
| `margins(m, type=ame)` | Average marginal effects |
| `hausman(m_fe, m_re)` | Hausman specification test |
| `irf(v, ...)` | Impulse response functions |
| `johansen(...)` | Johansen cointegration test |
| `testparm(m, vars)` | Joint significance test |
| `vif(m)` | Variance inflation factors |

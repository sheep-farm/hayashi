# All Estimators

Quick reference for all 46 estimators available in Hayashi.

## Cross-Section

| Command | Description | Syntax |
|---------|-------------|--------|
| `reg` | OLS linear regression | `reg(Y ~ X1 + X2, df)` |
| `iv` | Instrumental variables / 2SLS | `iv(Y ~ X_exog + X_endo, ~ Z + X_exog, df)` |
| `logit` | Logistic regression | `logit(Y ~ X1 + X2, df)` |
| `probit` | Probit regression | `probit(Y ~ X1 + X2, df)` |
| `ologit` | Ordered logit | `ologit(Y ~ X1 + X2, df)` |
| `oprobit` | Ordered probit | `oprobit(Y ~ X1 + X2, df)` |
| `mlogit` | Multinomial logit | `mlogit(Y ~ X1 + X2, df, base=1)` |
| `tobit` | Tobit (censored regression) | `tobit(Y ~ X1 + X2, df, ll=0)` |
| `truncreg` | Truncated regression | `truncreg(Y ~ X1 + X2, df, ll=0)` |
| `heckman` | Heckman selection model | `heckman(Y ~ X1, select: Z1 + Z2, df)` |
| `qreg` | Quantile regression | `qreg(Y ~ X1 + X2, df, q=0.5)` |
| `nbreg` | Negative binomial regression | `nbreg(Y ~ X1 + X2, df)` |
| `poisson` | Poisson regression | `poisson(Y ~ X1 + X2, df)` |
| `zip` | Zero-inflated Poisson | `zip(Y ~ X1, inflate: Z1 + Z2, df)` |

## Panel Data

| Command | Description | Syntax |
|---------|-------------|--------|
| `fe` | Fixed effects (within estimator) | `fe(Y ~ X1 + X2, df)` |
| `re` | Random effects (GLS) | `re(Y ~ X1 + X2, df)` |
| `be` | Between estimator | `be(Y ~ X1 + X2, df)` |
| `feiv` | FE with instrumental variables | `feiv(Y ~ X_exog + X_endo, ~ Z, df)` |
| `xtpoisson` | Panel Poisson | `xtpoisson(Y ~ X1 + X2, df, fe)` |
| `xtlogit` | Panel logit | `xtlogit(Y ~ X1 + X2, df, fe)` |
| `xtprobit` | Panel probit (RE only) | `xtprobit(Y ~ X1 + X2, df)` |
| `xtnbreg` | Panel negative binomial | `xtnbreg(Y ~ X1 + X2, df)` |
| `xttobit` | Panel tobit | `xttobit(Y ~ X1 + X2, df, ll=0)` |

## Time Series

| Command | Description | Syntax |
|---------|-------------|--------|
| `arima` | ARIMA / ARIMAX | `arima(Y, df, order=(p,d,q))` |
| `garch` | GARCH(p,q) | `garch(Y, df, order=(1,1))` |
| `egarch` | Exponential GARCH | `egarch(Y, df, order=(1,1))` |
| `gjrgarch` | GJR-GARCH (threshold) | `gjrgarch(Y, df, order=(1,1))` |
| `var` | Vector autoregression | `var(Y1 + Y2, df, lags=p)` |
| `vecm` | Vector error correction | `vecm(Y1 + Y2, df, lags=p, rank=r)` |
| `arch` | ARCH(q) | `arch(Y, df, order=q)` |
| `svar` | Structural VAR | `svar(Y1 + Y2, df, lags=p, type=short)` |

## Causal Inference

| Command | Description | Syntax |
|---------|-------------|--------|
| `did` | Difference-in-differences | `did(Y ~ X, df, treat=D, post=P)` |
| `rdd` | Regression discontinuity | `rdd(Y ~ X, df, cutoff=c, running=R)` |
| `synth` | Synthetic control | `synth(Y, df, treat_unit=id, treat_time=t)` |
| `psm` | Propensity score matching | `psm(Y ~ X1 + X2, df, treat=D)` |

## Finance

| Command | Description | Syntax |
|---------|-------------|--------|
| `fmb` | Fama-MacBeth regression | `fmb(ret ~ beta + size + bm, df, time=month)` |
| `dcc` | Dynamic conditional correlation | `dcc(Y1 + Y2, df)` |
| `mgarch` | Multivariate GARCH | `mgarch(Y1 + Y2, df, model=bekk)` |

## Robust / Semiparametric

| Command | Description | Syntax |
|---------|-------------|--------|
| `rreg` | Robust regression (M-estimator) | `rreg(Y ~ X1 + X2, df)` |
| `qreg` | Quantile regression | `qreg(Y ~ X1 + X2, df, q=0.5)` |
| `loess` | Local polynomial smoothing | `loess(Y ~ X, df, bw=0.3)` |
| `npregress` | Kernel nonparametric regression | `npregress(Y ~ X, df)` |

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
| `streg` | Parametric survival (Weibull, etc.) | `streg(T ~ X1 + X2, df, event=D, dist=weibull)` |

## Multivariate

| Command | Description | Syntax |
|---------|-------------|--------|
| `sureg` | Seemingly unrelated regressions | `sureg(eq1: Y1 ~ X1, eq2: Y2 ~ X2, df)` |
| `system3sls` | System three-stage least squares | `system3sls(eq1: Y1 ~ X1, eq2: Y2 ~ X2, df)` |

## Common Options

All estimators accept these where applicable:

| Option | Description |
|--------|-------------|
| `cov=robust` | Heteroskedasticity-robust SE (HC1) |
| `cov=hc0` ... `cov=hc4` | Specific HC variant |
| `cov=cluster(var)` | Cluster-robust SE |
| `cov=cluster(v1, v2)` | Two-way cluster SE |
| `nw=L` | Newey-West HAC SE with L lags |
| `if=(condition)` | Subsample estimation |
| `vce=bootstrap` | Bootstrap standard errors |

## Post-Estimation

| Command | Description |
|---------|-------------|
| `esttab(m1, m2, ...)` | Side-by-side estimation table |
| `predict(m, df, ...)` | Fitted values, residuals, probabilities |
| `margins(m, type=ame)` | Average marginal effects |
| `hausman(m_fe, m_re)` | Hausman specification test |
| `irf(v, ...)` | Impulse response functions |
| `johansen(...)` | Johansen cointegration test |
| `testparm(m, vars)` | Joint significance test |
| `linktest(m)` | Specification link test |
| `estat_vif(m)` | Variance inflation factors |

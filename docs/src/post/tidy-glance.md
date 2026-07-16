# Tidy & Glance

`tidy` and `glance` extract model results as DataFrames, following the
convention popularized by R's `broom` package. This makes it easy to
pipe estimation output into further data manipulation, export, or
visualization.

## tidy — coefficient table

```
let m = ols(Y ~ X1 + X2, df)
let t = tidy(m)
print(t)
```

Returns a DataFrame with one row per coefficient:

| Column | Description |
|---|---|
| `variable` | Coefficient name |
| `coef` | Point estimate |
| `std_err` | Standard error |
| `t` | t-statistic (or `z` for MLE models) |
| `p_value` | Two-sided p-value |
| `conf_low` | Lower 95% confidence bound |
| `conf_high` | Upper 95% confidence bound |

### Supported model types

`tidy` works with all estimators that report coefficients:

| Category | Models |
|---|---|
| Cross-section | OLS, WLS, IV, logit, probit, ologit, oprobit, mlogit, clogit, cmnlogit, cpoisson, tobit, heckman, qreg, nbreg, poisson, zip, zinb, rlm, glm, gee, betareg |
| Panel | FE, RE, BE, FE2SLS, PCSE, PanelGLS, GLSAR, Arellano-Bond, SystemGMM |
| Time series | ARIMA, AutoReg, ARDL, GARCH, EGARCH, GJR-GARCH, VAR, VECM, VARMA, ETS, LocalLevel, RecursiveLS |
| Causal | DiD, Threshold |
| Survival | Cox |
| Multivariate | SUR, 3SLS, Conditional, GAM, Mixed |
| Regularization | ridge, lasso, elasticnet |
| Other | Rolling, Penalized |

### Example: export coefficients

```
let m = ols(Y ~ X1 + X2, df, cov=robust)
let t = tidy(m)
export(t, "csv", "coefs.csv")
```

### Example: filter significant coefficients

```
let m = ols(Y ~ X1 + X2 + X3, df)
let t = tidy(m)
keep t if p_value < 0.05
print(t)
```

## glance — model fit statistics

```
let m = ols(Y ~ X1 + X2, df)
let g = glance(m)
print(g)
```

Returns a one-row DataFrame with model-level statistics. Available
keys vary by model type:

| Key | Description | Available for |
|---|---|---|
| `r2` | R-squared | OLS, IV, FE, RE, Panel, GLSAR, AutoReg, ARDL, DiD, Penalized |
| `adj_r2` | Adjusted R-squared | OLS, AutoReg, ARDL |
| `pseudo_r2` | McFadden pseudo R-squared | logit, probit, ordered, Poisson, NegBin, GLM, Quantile |
| `n` | Number of observations | all models |
| `f_stat` | F-statistic | OLS |
| `prob_f` | F-test p-value | OLS |
| `aic` | Akaike information criterion | OLS, ARIMA, AutoReg, ARDL, GARCH, VAR, GLM, Poisson, NegBin, Tobit, Ordered, Conditional, ZeroInflated, Mixed, ETS, Cox |
| `bic` | Bayesian information criterion | same as AIC |
| `log_lik` | Log-likelihood | OLS, ARIMA, GARCH, GLM, Poisson, NegBin, Tobit, Ordered, Cox, Conditional, ZeroInflated, Mixed, LocalLevel |
| `sigma` | Residual standard error | OLS, IV, FE, PCSE, PanelGLS, FE2SLS |
| `sigma2` | Innovation variance | ARIMA |
| `sigma_u` | Between-entity SD | RE |
| `sigma_e` | Within-entity SD | RE |
| `theta` | RE theta | RE |
| `j_stat` | Hansen J statistic | GMM |
| `j_p_value` | Hansen J p-value | GMM |
| `df_overid` | Overidentification df | GMM |
| `sargan_stat` | Sargan statistic | SystemGMM |
| `sargan_p` | Sargan p-value | SystemGMM |
| `alpha` | NegBin dispersion | NegBin, ZeroInflated |
| `rho` | Heckman rho | Heckman |
| `delta` | Heckman delta | Heckman |
| `tau` | Quantile | Quantile |
| `concordance` | C-index | Cox |
| `gcv` | GCV score | GAM |
| `att` | Average treatment effect | DiD |
| `threshold` | Threshold parameter | Threshold |
| `rank` | Cointegration rank | VECM |
| `n_entities` | Number of entities | FE |
| `n_groups` | Number of groups | Mixed, GEE |
| `n_censored` | Censored observations | Tobit |
| `deviance` | Deviance | GLM |
| `sse` | Sum of squared errors | ETS |
| `sigma_obs` | Observation noise SD | LocalLevel |
| `sigma_state` | State noise SD | LocalLevel |

### Example: compare models

```
let m1 = ols(Y ~ X1, df)
let m2 = ols(Y ~ X1 + X2, df)
let g1 = glance(m1)
let g2 = glance(m2)
print(g1)
print(g2)
```

## Notes

- Both `tidy` and `glance` return DataFrames, so they can be
  exported, filtered, or piped into any DataFrame operation.
- For models without standard errors (e.g. RecursiveLS, Threshold),
  `tidy` reports zeros for `std_err`, `t`, and `p_value`.
- `glance` keys are model-specific: only relevant statistics are
  reported for each model type. Use `names(glance(m))` to inspect
  available keys.

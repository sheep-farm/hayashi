# Hypothesis Tests

## Wald test on a single coefficient

```
let m = ols(Y ~ X1 + X2 + X3, df)

test(m, "X1")
```

Reports the t-statistic, p-value, and 95% CI for H0: beta_X1 = 0.

## Joint test on multiple coefficients

```
test(m, "X1", "X2")
```

F-test for H0: beta_X1 = beta_X2 = 0 simultaneously.

## Linear restriction

```
test(m, "X1 = X2")
```

Tests H0: beta_X1 = beta_X2. Accepts any linear combination of coefficients.

## Breusch-Pagan

```
test(m, "bp")
```

Tests for heteroskedasticity by regressing squared residuals on the regressors.

## White

```
test(m, "white")
```

General heteroskedasticity test using all regressors, their squares, and cross products.

## Durbin-Watson

```
test(m, "dw")
```

Tests for first-order autocorrelation in OLS residuals. Reports the DW statistic and approximate p-value.

## RESET (Ramsey)

```
reset(m)
```

Regression Specification Error Test. Adds powers of fitted values and tests their joint significance. Separate function because it re-estimates the model internally.

## Jarque-Bera

```
jb(m)
```

Tests normality of residuals via skewness and kurtosis. Reports the JB statistic and p-value.

## Panel diagnostics

### Breusch-Pagan LM (RE vs OLS)

```
bptest(df, Y ~ X1 + X2, id="entity")
```

Tests H0: σ²_u = 0 (no panel effect — pooled OLS adequate). If rejected, use RE or FE instead of pooled OLS. Requires `id=` column or prior `xtset`.

### F-test for fixed effects (FE vs OLS)

```
ftest_fe(df, Y ~ X1 + X2, id="entity")
```

Tests H0: all individual effects are zero (pooled OLS adequate). If rejected, use FE. Reports SSR pooled, SSR FE, and the F-statistic.

### Hausman (FE vs RE)

```
hausman(m_fe, m_re)
```

Tests H0: RE is consistent (individual effects uncorrelated with regressors). If rejected, use FE. Requires both an FE and an RE model.

### Robust Hausman (FE vs RE)

```
hausman_robust(m_fe, m_re)
```

Robust version of the Hausman test (Cameron & Trivedi 2005; Wooldridge 2010). Uses the robust covariance of the coefficient differences instead of the classical variance comparison. Valid under heteroskedasticity and clustering — situations where the standard Hausman test is unreliable.

Reports the Chi-squared statistic, degrees of freedom, p-value, and the per-variable beta differences (FE minus RE). Alias: `hausman_r`.

```
xtset(panel, firm, year)
let mfe = fe(output ~ capital + labor, panel)
let mre = re(output ~ capital + labor, panel)
hausman_robust(mfe, mre)
```

### Wooldridge serial correlation

```
wooldridge(df, Y ~ X1, id="entity", time="time")
```

Tests H0: no first-order serial correlation in idiosyncratic errors. Useful for panel time series.

### Pesaran CD (cross-sectional dependence)

```
pesaran(df, Y ~ X1, id="entity", time="time")
```

Tests H0: no cross-sectional dependence. Relevant for large-N panels where spatial or network spillovers may exist.

### Arellano-Bond m1/m2

```
abtest(df, Y ~ X1, id="entity", time="time")
```

Tests for serial correlation in first-differenced residuals. m1 should reject H0 (FD induces AR(1) by construction); m2 should not reject H0 (validates instruments y_{i,t-2} for GMM).

### Mundlak

```
mundlak(df, Y ~ X1 + X2, id="entity")
```

Tests H0: RE is consistent, by augmenting the model with entity means of regressors. A generalization of the Hausman test.

## Likelihood-ratio test

```
lrtest(m_restricted, m_unrestricted)
```

Tests H0: the restricted model is adequate (the additional parameters in the unrestricted model are zero). The LR statistic is:

LR = -2 * (ln L_restricted - ln L_unrestricted) ~ chi²(df)

where df = k_unrestricted - k_restricted. The models must be nested (the restricted model is a special case of the unrestricted one).

Supports any estimator that reports a log-likelihood: OLS, logit/probit, Poisson, NegBin, Tobit, Ordered, Mixed, Zero-Inflated, GLM, GARCH, ARIMA.

```
let m1 = ols(Y ~ X1, df)
let m2 = ols(Y ~ X1 + X2, df)
lrtest(m1, m2)
```

## Robust F-test (Wald)

```
ftest_robust(m)
ftest_robust(m, vars="x1,x2")
```

Robust joint significance test (Wooldridge 2010). Tests H0: all specified coefficients are jointly zero, using a Wald statistic built from the robust covariance matrix. Valid under heteroskedasticity and clustering.

By default, tests all slope coefficients (excludes the intercept). Use `vars=` to test a subset:

```
let m = ols(y ~ x1 + x2 + x3, df, cov=robust)
ftest_robust(m, vars="x1,x2")
```

Reports the Wald Chi-squared statistic, F-statistic, and both p-values (F and Chi2). Supports OLS, FE, and RE models. Alias: `f_robust`.

## Notes

- `test` works with any estimator that stores a variance-covariance matrix (OLS, IV, logit, probit, panel models, etc.).
- Diagnostic tests (`bp`, `white`, `dw`) are restricted to OLS results.
- Panel diagnostics (`bptest`, `ftest_fe`, `wooldridge`, `pesaran`, `abtest`, `mundlak`) require `id=` and `time=` columns, or a prior `xtset(df, id, time)`.
- All tests print a summary table to stdout and return a record with fields `stat`, `pval`, and `df`.

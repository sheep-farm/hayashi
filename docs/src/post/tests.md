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

## Notes

- `test` works with any estimator that stores a variance-covariance matrix (OLS, IV, logit, probit, panel models, etc.).
- Diagnostic tests (`bp`, `white`, `dw`) are restricted to OLS results.
- All tests print a summary table to stdout and return a record with fields `stat`, `pval`, and `df`.

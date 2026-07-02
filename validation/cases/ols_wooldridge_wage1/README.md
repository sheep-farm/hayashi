:
# OLS wage equation on Wooldridge `wage1`

This validation case estimates a standard Mincer-style wage equation:

```
wage ~ educ + exper + tenure
```

using the `wage1` dataset from the Wooldridge package.

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 526 observations × 24 variables.

## Reference implementation

- **R:** `lm(wage ~ educ + exper + tenure, data = wage1)`
- **Python statsmodels:** `smf.ols("wage ~ educ + exper + tenure", data=wage1)`
- **Stata:** `reg wage educ exper tenure` (optional, if available)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)
- R-squared
- number of observations

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |
| r_squared | 1e-8 | Fit statistic is a simple scalar |
| nobs | 0 | Exact count |

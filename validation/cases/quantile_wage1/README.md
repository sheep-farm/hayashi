# Quantile wage regression on Wooldridge `wage1`

This validation case estimates a median quantile regression of wage on education, experience, and tenure.

## Model

```
wage ~ educ + exper + tenure
```

at the median (tau = 0.5).

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 526 observations × 24 variables.

## Reference implementation

- **R:** `rq(wage ~ educ + exper + tenure, data = wage1, tau = 0.5)`
- **Python:** `statsmodels.quantreg("wage ~ educ + exper + tenure", data = wage1).fit(q=0.5)`
- **Hayashi:** `qreg(wage ~ educ + exper + tenure, df, q=0.5)`

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | Quantile regression algorithms may differ slightly |
| standard_errors | 1e-4 | Same tolerance as coefficients |

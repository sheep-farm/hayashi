# Robust linear model on Wooldridge `wage1`

This validation case estimates a Huber robust linear regression of log wage on education, experience, and tenure.

## Model

```
lwage ~ educ + exper + tenure
```

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **R:** `MASS::rlm(lwage ~ educ + exper + tenure, data = wage1)`
- **Python:** `statsmodels.formula.rlm("lwage ~ educ + exper + tenure", data=df, M=HuberT()).fit()`
- **Hayashi:** `rlm(lwage ~ educ + exper + tenure, df)`

## Compared quantities

- Regression coefficients and standard errors for `const`, `educ`, `exper`, `tenure`.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | Iteratively reweighted least squares may differ slightly |
| standard_errors | 1e-2 | SE formulas differ between implementations |

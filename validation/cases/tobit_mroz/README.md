# Tobit hours-worked model on Wooldridge `mroz`

This validation case estimates a Tobit regression of hours worked, left-censored at zero.

## Model

```
hours ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6
```

with left censoring at `hours = 0`.

## Dataset

- **Name:** `wooldridge::mroz`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 753 observations × 22 variables.

## Reference implementation

- **R:** `AER::tobit(hours ~ ..., data = mroz, left = 0)`
- **Python:** Custom maximum-likelihood Tobit implementation via `scipy.optimize` (log-likelihood with normal censoring).
- **Hayashi:** `tobit(hours ~ ..., df, ll=0)`

## Compared quantities

- coefficients
- standard errors (observed inverse-Hessian in Python, AER's default in R)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | MLE should converge to the same maximum |
| standard_errors | 1e-3 | Allow small numerical differences in covariance estimation |

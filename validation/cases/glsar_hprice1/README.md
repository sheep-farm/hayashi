# GLSAR(1) on Wooldridge `hprice1`

This validation case estimates a linear model with AR(1) errors on housing price data.

## Model

```
price ~ lotsize + sqrft + bdrms
```

with first-order autoregressive errors.

## Dataset

- **Name:** `wooldridge::hprice1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 88 observations × 11 variables.

## Reference implementation

- **Python:** `statsmodels.regression.linear_model.GLSAR.from_formula(..., rho=1).fit()`
- **Hayashi:** `glsar(price ~ lotsize + sqrft + bdrms, df, order=1)`
- **R:** explicit base-R iterative GLSAR(1) using adjusted Yule-Walker updates

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-1 | Different iterative convergence tolerances between packages |
| standard_errors | 1e-1 | Same tolerance as coefficients |

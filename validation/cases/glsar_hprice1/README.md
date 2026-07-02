# GLSAR(1) on Wooldridge `hprice1`

This validation case estimates a linear model with AR(1) errors on housing price data.

## Status

`blocked` — Hayashi and `statsmodels.regression.linear_model.GLSAR` converge to different AR(1) `rho` estimates, so the coefficients differ beyond reasonable tolerance. A common reference implementation would be needed before this case can be validated.

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
- **R:** fallback to OLS because dedicated AR-error GLS packages are not guaranteed in the environment.

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | GLSAR should match OLS coefficients closely when autocorrelation is mild |
| standard_errors | 1e-3 | Same tolerance as coefficients |

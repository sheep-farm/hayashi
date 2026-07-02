# WLS housing price equation on Wooldridge `hprice1`

This validation case estimates a weighted least squares (WLS) model of
housing prices on lot size, square footage, and number of bedrooms.

## Model

```
price ~ lotsize + sqrft + bdrms
```

with weights `w = 1 / lotsize`, reflecting the assumption that error variance
is proportional to lot size.

## Dataset

- **Name:** `wooldridge::hprice1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 88 observations × 10 variables.

## Reference implementation

- **R:** `lm(price ~ lotsize + sqrft + bdrms, data = hprice1, weights = w)`
- **Python statsmodels:** `smf.wls("price ~ lotsize + sqrft + bdrms", data = hprice1, weights = hprice1["w"])`
- **Hayashi:** `wls(price ~ lotsize + sqrft + bdrms, df, weights = "w")`

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | WLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

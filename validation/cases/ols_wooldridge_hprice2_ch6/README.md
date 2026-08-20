# OLS log housing price equation with rooms quadratic on Wooldridge hprice2 (Chapter 6, Example 6.2)

This validation case estimates the log housing price equation with quadratic in rooms from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 6, Example 6.2:

```
lprice ~ lnox + ldist + rooms + roomsq + stratio
```

## Dataset

- **Name:** `wooldridge::hprice2`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 506 observations × 12 variables.

## Reference implementation

- **R:** `lm(lprice ~ lnox + ldist + rooms + roomsq + stratio, data = hprice2)`
- **Python statsmodels:** `smf.ols("lprice ~ 1 + lnox + ldist + rooms + roomsq + stratio", data=hprice2).fit()`
- **Stata:** `reg lprice lnox ldist rooms roomsq stratio` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

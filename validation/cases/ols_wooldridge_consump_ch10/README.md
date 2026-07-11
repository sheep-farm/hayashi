# OLS consumption growth on income growth on Wooldridge consump (Chapter 10, Example 10.4)

This validation case estimates the consumption growth on income growth equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 10, Example 10.4:

```
gc ~ gy
```

## Dataset

- **Name:** `wooldridge::consump`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 37 observations × 24 variables.

## Reference implementation

- **R:** `lm(gc ~ gy, data = consump)`
- **Python statsmodels:** `smf.ols("gc ~ 1 + gy", data=consump).fit()`
- **Stata:** `reg gc gy` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

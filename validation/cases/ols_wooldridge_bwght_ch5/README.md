# OLS birth weight equation on Wooldridge bwght (Chapter 5, Example 5.2)

This validation case estimates the birth weight and maternal smoking equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 5, Example 5.2:

```
lbwght ~ cigs + lfaminc
```

## Dataset

- **Name:** `wooldridge::bwght`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 1,388 observations × 14 variables.

## Reference implementation

- **R:** `lm(lbwght ~ cigs + lfaminc, data = bwght)`
- **Python statsmodels:** `smf.ols("lbwght ~ 1 + cigs + lfaminc", data=bwght).fit()`
- **Stata:** `reg lbwght cigs lfaminc` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

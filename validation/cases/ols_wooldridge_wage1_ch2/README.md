# OLS log wage equation on Wooldridge `wage1` (Chapter 2, Example 2.10)

This validation case estimates the first textbook wage equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 2, Example 2.10:

```
lwage ~ educ
```

where `lwage` is the natural logarithm of average hourly earnings and `educ` is years of education.

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 526 observations × 24 variables.

## Reference implementation

- **R:** `lm(lwage ~ educ, data = wage1)`
- **Python statsmodels:** `smf.ols("lwage ~ 1 + educ", data=wage1).fit()`
- **Stata:** `reg lwage educ` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

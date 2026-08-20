# OLS returns to two- and four-year colleges on Wooldridge twoyear (Chapter 4, Example 4.10)

This validation case estimates the returns to two-year and four-year college credits from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 4, Example 4.10:

```
lwage ~ jc + univ + exper
```

## Dataset

- **Name:** `wooldridge::twoyear`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 6,763 observations × 23 variables.

## Reference implementation

- **R:** `lm(lwage ~ jc + univ + exper, data = twoyear)`
- **Python statsmodels:** `smf.ols("lwage ~ 1 + jc + univ + exper", data=twoyear).fit()`
- **Stata:** `reg lwage jc univ exper` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

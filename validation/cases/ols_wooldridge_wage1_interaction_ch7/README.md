# OLS hourly wage equation with marriage-gender interactions on Wooldridge wage1 (Chapter 7, Example 7.6)

This validation case estimates the hourly wage equation with marriage and gender dummy interactions from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 7, Example 7.6:

```
lwage ~ marrmale + marrfem + singfem + educ + exper + expersq + tenure + tenursq
```

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 526 observations × 24 variables.

## Reference implementation

- **R:** `lm(lwage ~ marrmale + marrfem + singfem + educ + exper + expersq + tenure + tenursq, data = wage1)`
- **Python statsmodels:** `smf.ols("lwage ~ 1 + marrmale + marrfem + singfem + educ + exper + expersq + tenure + tenursq", data=wage1).fit()`
- **Stata:** `reg lwage marrmale marrfem singfem educ exper expersq tenure tenursq` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

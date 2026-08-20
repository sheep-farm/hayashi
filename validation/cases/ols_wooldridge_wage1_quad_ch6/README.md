# OLS wage equation with experience quadratic on Wooldridge wage1 (Chapter 6, Section 6.2)

This validation case estimates the wage equation with a quadratic in experience from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 6, Section 6.2:

```
wage ~ exper + expersq
```

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 526 observations × 24 variables.

## Reference implementation

- **R:** `lm(wage ~ exper + expersq, data = wage1)`
- **Python statsmodels:** `smf.ols("wage ~ 1 + exper + expersq", data=wage1).fit()`
- **Stata:** `reg wage exper expersq` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

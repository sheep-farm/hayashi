# OLS difference-in-differences housing price equation on Wooldridge kielmc (Chapter 13, Example 13.1)

This validation case estimates the difference-in-differences housing price equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 13, Example 13.1:

```
lprice ~ y81 + nearinc + y81nrinc
```

## Dataset

- **Name:** `wooldridge::kielmc`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 321 observations × 25 variables.

## Reference implementation

- **R:** `lm(lprice ~ y81 + nearinc + y81nrinc, data = kielmc)`
- **Python statsmodels:** `smf.ols("lprice ~ 1 + y81 + nearinc + y81nrinc", data=kielmc).fit()`
- **Stata:** `reg lprice y81 nearinc y81nrinc` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

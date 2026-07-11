# OLS CEO salary on return on equity on Wooldridge ceosal1 (Chapter 2, Example 2.3)

This validation case estimates the CEO salary and return on equity equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 2, Example 2.3:

```
salary ~ roe
```

## Dataset

- **Name:** `wooldridge::ceosal1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 209 observations × 12 variables.

## Reference implementation

- **R:** `lm(salary ~ roe, data = ceosal1)`
- **Python statsmodels:** `smf.ols("salary ~ 1 + roe", data=ceosal1).fit()`
- **Stata:** `reg salary roe` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

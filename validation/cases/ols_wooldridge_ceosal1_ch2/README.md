# OLS CEO salary equation on Wooldridge ceosal1 (Chapter 2, Example 2.11)

This validation case estimates the log-log CEO salary equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 2, Example 2.11:

```
lsalary ~ lsales
```

## Dataset

- **Name:** `wooldridge::ceosal1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 209 observations × 12 variables.

## Reference implementation

- **R:** `lm(lsalary ~ lsales, data = ceosal1)`
- **Python statsmodels:** `smf.ols("lsalary ~ 1 + lsales", data=ceosal1).fit()`
- **Stata:** `reg lsalary lsales` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

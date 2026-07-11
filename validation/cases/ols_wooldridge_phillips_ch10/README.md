# OLS static Phillips curve on Wooldridge phillips (Chapter 10, Example 10.1)

This validation case estimates the static Phillips curve from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 10, Example 10.1:

```
inf ~ unem
```

## Dataset

- **Name:** `wooldridge::phillips`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 56 observations × 7 variables.

## Reference implementation

- **R:** `lm(inf ~ unem, data = phillips)`
- **Python statsmodels:** `smf.ols("inf ~ 1 + unem", data=phillips).fit()`
- **Stata:** `reg inf unem` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

# OLS expectations-augmented Phillips curve on Wooldridge phillips (Chapter 11, Example 11.5)

This validation case estimates the expectations-augmented Phillips curve from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 11, Example 11.5:

```
cinf ~ unem
```

## Dataset

- **Name:** `wooldridge::phillips`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 56 observations × 7 variables.

## Reference implementation

- **R:** `lm(cinf ~ unem, data = phillips)`
- **Python statsmodels:** `smf.ols("cinf ~ 1 + unem", data=phillips).fit()`
- **Stata:** `reg cinf unem` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

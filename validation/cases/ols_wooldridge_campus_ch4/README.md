# OLS campus crime equation on Wooldridge campus (Chapter 4, Example 4.4)

This validation case estimates the log-log campus crime and enrollment equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 4, Example 4.4:

```
lcrime ~ lenroll
```

## Dataset

- **Name:** `wooldridge::campus`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 97 observations × 7 variables.

## Reference implementation

- **R:** `lm(lcrime ~ lenroll, data = campus)`
- **Python statsmodels:** `smf.ols("lcrime ~ 1 + lenroll", data=campus).fit()`
- **Stata:** `reg lcrime lenroll` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

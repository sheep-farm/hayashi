# OLS attendance effects on exam score on Wooldridge attend (Chapter 6, Example 6.3)

This validation case estimates the attendance effects on standardized final exam score from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 6, Example 6.3:

```
stndfnl ~ atndrte + priGPA + ACT
```

## Dataset

- **Name:** `wooldridge::attend`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 680 observations × 11 variables.

## Reference implementation

- **R:** `lm(stndfnl ~ atndrte + priGPA + ACT, data = attend)`
- **Python statsmodels:** `smf.ols("stndfnl ~ 1 + atndrte + priGPA + ACT", data=attend).fit()`
- **Stata:** `reg stndfnl atndrte priGPA ACT` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

# OLS log hourly wage equation with female dummy on Wooldridge wage1 (Chapter 7, Example 7.1)

This validation case estimates the log hourly wage equation with a qualitative dummy variable from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 7, Example 7.1:

```
lwage ~ female + educ + exper + tenure
```

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 526 observations × 24 variables.

## Reference implementation

- **R:** `lm(lwage ~ female + educ + exper + tenure, data = wage1)`
- **Python statsmodels:** `smf.ols("lwage ~ 1 + female + educ + exper + tenure", data=wage1).fit()`
- **Stata:** `reg lwage female educ exper tenure` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

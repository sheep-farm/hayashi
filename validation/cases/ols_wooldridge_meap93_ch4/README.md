# OLS math pass rate equation on Wooldridge meap93 (Chapter 4, Examples 4.2/4.10)

This validation case estimates the math pass rate equation with log salary, staff and enrollment from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 4, Examples 4.2 and 4.10:

```
math10 ~ ltotcomp + lstaff + lenroll
```

## Dataset

- **Name:** `wooldridge::meap93`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 408 observations × 17 variables.

## Reference implementation

- **R:** `lm(math10 ~ ltotcomp + lstaff + lenroll, data = meap93)`
- **Python statsmodels:** `smf.ols("math10 ~ 1 + ltotcomp + lstaff + lenroll", data=meap93).fit()`
- **Stata:** `reg math10 ltotcomp lstaff lenroll` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

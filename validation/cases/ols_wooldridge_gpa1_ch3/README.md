# OLS college GPA equation on Wooldridge `gpa1` (Chapter 3, Example 3.1)

This validation case estimates the college GPA equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 3, Example 3.1:

```
colGPA ~ hsGPA + ACT
```

## Dataset

- **Name:** `wooldridge::gpa1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 141 observations × 29 variables.

## Reference implementation

- **R:** `lm(colGPA ~ hsGPA + ACT, data = gpa1)`
- **Python statsmodels:** `smf.ols("colGPA ~ 1 + hsGPA + ACT", data=gpa1).fit()`
- **Stata:** `reg colGPA hsGPA ACT` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

# OLS sleep equation on Wooldridge sleep75 (Chapter 5, Problem 3.3 / Example 5.3)

This validation case estimates the sleep-work tradeoff equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 5, Problem 3.3:

```
sleep ~ totwrk + educ + age
```

## Dataset

- **Name:** `wooldridge::sleep75`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 706 observations × 34 variables.

## Reference implementation

- **R:** `lm(sleep ~ totwrk + educ + age, data = sleep75)`
- **Python statsmodels:** `smf.ols("sleep ~ 1 + totwrk + educ + age", data=sleep75).fit()`
- **Stata:** `reg sleep totwrk educ age` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

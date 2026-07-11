# OLS education equation on Wooldridge htv (Chapter 9, Example 9.3)

This validation case estimates the education and parental education/ability equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 9, Example 9.3:

```
educ ~ motheduc + fatheduc + abil
```

## Dataset

- **Name:** `wooldridge::htv`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 1,230 observations × 23 variables.

## Reference implementation

- **R:** `lm(educ ~ motheduc + fatheduc + abil, data = htv)`
- **Python statsmodels:** `smf.ols("educ ~ 1 + motheduc + fatheduc + abil", data=htv).fit()`
- **Stata:** `reg educ motheduc fatheduc abil` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

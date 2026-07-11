# OLS election outcomes on campaign share on Wooldridge vote1 (Chapter 2, Examples 2.5/2.9)

This validation case estimates the election outcomes and campaign expenditure share equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 2, Examples 2.5 and 2.9:

```
voteA ~ shareA
```

## Dataset

- **Name:** `wooldridge::vote1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 173 observations × 10 variables.

## Reference implementation

- **R:** `lm(voteA ~ shareA, data = vote1)`
- **Python statsmodels:** `smf.ols("voteA ~ 1 + shareA", data=vote1).fit()`
- **Stata:** `reg voteA shareA` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

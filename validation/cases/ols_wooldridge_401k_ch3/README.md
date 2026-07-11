# OLS 401(k) participation equation on Wooldridge `401k` (Chapter 3, Example 3.3)

This validation case estimates the 401(k) pension plan participation equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 3, Example 3.3:

```
prate ~ mrate + age
```

## Dataset

- **Name:** `wooldridge::401k`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 1,534 observations × 8 variables.

## Reference implementation

- **R:** `lm(prate ~ mrate + age, data = k401k)`
- **Python statsmodels:** `smf.ols("prate ~ 1 + mrate + age", data=k401k).fit()`
- **Stata:** `reg prate mrate age` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

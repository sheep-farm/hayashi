# OLS pooled job training scrap equation on Wooldridge jtrain (Chapter 14, Example 14.3)

This validation case estimates the pooled job training scrap rate equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 14, Example 14.3:

```
lscrap ~ d88 + d89 + grant + grant_1 + lsales + lemploy
```

## Dataset

- **Name:** `wooldridge::jtrain`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 471 observations × 30 variables.

## Reference implementation

- **R:** `lm(lscrap ~ d88 + d89 + grant + grant_1 + lsales + lemploy, data = jtrain)`
- **Python statsmodels:** `smf.ols("lscrap ~ 1 + d88 + d89 + grant + grant_1 + lsales + lemploy", data=jtrain).fit()`
- **Stata:** `reg lscrap d88 d89 grant grant_1 lsales lemploy` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

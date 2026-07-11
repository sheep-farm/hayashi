# OLS fertility distributed lag equation on Wooldridge fertil3 (Chapter 13, Example 13.3)

This validation case estimates the fertility distributed lag equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 13, Example 13.3:

```
gfr ~ pe + pe_1 + pe_2 + ww2 + pill
```

## Dataset

- **Name:** `wooldridge::fertil3`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 72 observations × 24 variables.

## Reference implementation

- **R:** `lm(gfr ~ pe + pe_1 + pe_2 + ww2 + pill, data = fertil3)`
- **Python statsmodels:** `smf.ols("gfr ~ 1 + pe + pe_1 + pe_2 + ww2 + pill", data=fertil3).fit()`
- **Stata:** `reg gfr pe pe_1 pe_2 ww2 pill` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

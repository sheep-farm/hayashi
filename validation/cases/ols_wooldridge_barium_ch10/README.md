# OLS barium chloride import equation on Wooldridge barium (Chapter 10, Example 10.5)

This validation case estimates the barium chloride import demand and antidumping filings from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 10, Example 10.5:

```
lchnimp ~ lchempi + lgas + lrtwex + befile6 + affile6 + afdec6
```

## Dataset

- **Name:** `wooldridge::barium`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 131 observations × 31 variables.

## Reference implementation

- **R:** `lm(lchnimp ~ lchempi + lgas + lrtwex + befile6 + affile6 + afdec6, data = barium)`
- **Python statsmodels:** `smf.ols("lchnimp ~ 1 + lchempi + lgas + lrtwex + befile6 + affile6 + afdec6", data=barium).fit()`
- **Stata:** `reg lchnimp lchempi lgas lrtwex befile6 affile6 afdec6` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

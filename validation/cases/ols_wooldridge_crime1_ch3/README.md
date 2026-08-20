# OLS arrest records equation on Wooldridge `crime1` (Chapter 3, Example 3.5)

This validation case estimates the arrest records equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 3, Example 3.5:

```
narr86 ~ pcnv + ptime86 + qemp86
```

## Dataset

- **Name:** `wooldridge::crime1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 2,725 observations × 16 variables.

## Reference implementation

- **R:** `lm(narr86 ~ pcnv + ptime86 + qemp86, data = crime1)`
- **Python statsmodels:** `smf.ols("narr86 ~ 1 + pcnv + ptime86 + qemp86", data=crime1).fit()`
- **Stata:** `reg narr86 pcnv ptime86 qemp86` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

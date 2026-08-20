# OLS Puerto Rican employment equation on Wooldridge prminwge (Chapter 10, Example 10.3)

This validation case estimates the Puerto Rican employment and minimum wage equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 10, Example 10.3:

```
lprepop ~ lmincov + lusgnp
```

## Dataset

- **Name:** `wooldridge::prminwge`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 38 observations × 25 variables.

## Reference implementation

- **R:** `lm(lprepop ~ lmincov + lusgnp, data = prminwge)`
- **Python statsmodels:** `smf.ols("lprepop ~ 1 + lmincov + lusgnp", data=prminwge).fit()`
- **Stata:** `reg lprepop lmincov lusgnp` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

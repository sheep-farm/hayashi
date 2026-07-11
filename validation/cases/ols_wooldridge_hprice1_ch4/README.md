# OLS log housing price equation on Wooldridge hprice1 (Chapter 4, Section 4.5)

This validation case estimates the log housing price equation with qualitative information from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 4, Section 4.5:

```
lprice ~ llotsize + lsqrft + bdrms + colonial
```

## Dataset

- **Name:** `wooldridge::hprice1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 88 observations × 10 variables.

## Reference implementation

- **R:** `lm(lprice ~ llotsize + lsqrft + bdrms + colonial, data = hprice1)`
- **Python statsmodels:** `smf.ols("lprice ~ 1 + llotsize + lsqrft + bdrms + colonial", data=hprice1).fit()`
- **Stata:** `reg lprice llotsize lsqrft bdrms colonial` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

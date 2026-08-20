# OLS efficient markets hypothesis on Wooldridge nyse (Chapter 11, Example 11.4)

This validation case estimates the AR(1) test of efficient markets hypothesis from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 11, Example 11.4:

```
ret ~ return_1
```

Note: the original Wooldridge variable `return` is renamed to `ret` in the generated CSV to avoid the Hayashi/Python reserved keyword.

## Dataset

- **Name:** `wooldridge::nyse`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 691 observations × 8 variables.

## Reference implementation

- **R:** `lm(ret ~ return_1, data = nyse_renamed)`
- **Python statsmodels:** `smf.ols("ret ~ 1 + return_1", data=df).fit()`
- **Stata:** `reg ret return_1` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

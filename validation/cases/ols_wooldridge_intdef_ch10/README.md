# OLS interest rates on inflation and deficits on Wooldridge intdef (Chapter 10, Example 10.2)

This validation case estimates the interest rate, inflation and deficit equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 10, Example 10.2:

```
i3 ~ inflation + deficit
```

Note: the original Wooldridge variables `inf` and `def` are renamed to `inflation` and `deficit` in the generated CSV to avoid reserved literals in Python/patsy.

## Dataset

- **Name:** `wooldridge::intdef`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 56 observations × 13 variables.

## Reference implementation

- **R:** `lm(i3 ~ inflation + deficit, data = intdef_renamed)`
- **Python statsmodels:** `smf.ols("i3 ~ 1 + inflation + deficit", data=df).fit()`
- **Stata:** `reg i3 inflation deficit` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-6 | OLS closed-form should match to machine precision |
| standard_errors | 1e-6 | Same precision target as coefficients |

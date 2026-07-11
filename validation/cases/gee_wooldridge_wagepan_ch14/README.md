# GEE wage equation on Wooldridge wagepan (Chapter 14, Example 14.4)

This validation case estimates the generalized estimating equations wage equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 14, Example 14.4:

```
lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87
```

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 4,360 observations × 44 variables.

## Reference implementation

- **R:** `geeglm(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87, id = nr, data = wagepan, family = gaussian, corstr = "independence")`
- **Python statsmodels:** `gee("lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87", data=wagepan, groups=wagepan["nr"], cov_struct=Independence(), family=Gaussian()).fit()`
- **Stata:** `xtgee lwage union married d81-d87, i(nr)` (optional)

## Compared quantities

- coefficients
- standard errors (robust/sandwich)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | GEE coefficients should match closely |
| standard_errors | 1e-4 | Sandwich SEs should match closely |

# Mixed linear model wage equation on Wooldridge wagepan (Chapter 14, Example 14.4)

This validation case estimates the mixed linear model wage equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 14, Example 14.4:

```
lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87
```

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 4,360 observations × 44 variables.

## Reference implementation

- **R:** `lmer(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87 + (1 | nr), data = wagepan, REML = TRUE)`
- **Python statsmodels:** `mixedlm("lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87", data=wagepan, groups=wagepan["nr"]).fit()`
- **Stata:** `mixed lwage union married d81-d87 || nr:` (optional)

## Compared quantities

- coefficients (fixed effects)
- standard errors

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | REML/ML fixed effects should match closely |
| standard_errors | 1e-4 | Same precision target |

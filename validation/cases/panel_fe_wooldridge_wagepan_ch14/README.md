# Panel fixed-effects wage equation on Wooldridge wagepan (Chapter 14, Example 14.4)

This validation case estimates the panel fixed-effects wage equation from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 14, Example 14.4:

```
lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87
```

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 4,360 observations × 44 variables.

## Reference implementation

- **R:** `plm(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87, data = wagepan, model = "within", index = c("nr", "year"))`
- **Python linearmodels:** `PanelOLS.from_formula("lwage ~ 1 + union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87 + EntityEffects", data=wagepan.set_index(["nr","year"])).fit()`
- **Stata:** `xtreg lwage union married d81-d87, fe` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | Fixed-effects closed-form should match closely |
| standard_errors | 1e-4 | Same precision target as coefficients |

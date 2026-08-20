# Random-effects investment demand on Grunfeld

This validation case estimates a panel random-effects investment demand model.

## Model

```
inv ~ value + capital
```

with random firm effects.

## Dataset

- **Name:** `grunfeld`
- **Source:** `statsmodels.datasets` (Python) and `plm` (R).
- **Licence:** Public teaching dataset.
- **Size:** 200 observations (10 firms × 20 years) × 5 variables.

## Reference implementation

- **R:** `plm(inv ~ value + capital, data = Grunfeld, index = c("firm", "year"), model = "random")`
- **Python:** `linearmodels.RandomEffects.from_formula("inv ~ 1 + value + capital", ...)`
- **Hayashi:** `xtset(df, firm, year)` then `re(inv ~ value + capital, df)`

The explicit `1 +` in the Python formula is required because `linearmodels`
does not add an intercept by default. The R and Hayashi formulas include one.

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | Random-effects GLS should match to high precision |
| standard_errors | 1e-4 | Same tolerance as coefficients |

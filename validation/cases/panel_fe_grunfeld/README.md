# Fixed-effects investment demand on Wooldridge `grunfeld`

This validation case estimates a panel fixed-effects investment demand model.

## Model

```
inv ~ value + capital
```

with firm fixed effects.

## Dataset

- **Name:** `wooldridge::grunfeld`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 200 observations (10 firms × 20 years) × 5 variables.

## Reference implementation

- **R:** `plm(inv ~ value + capital, data = grunfeld, index = c("firm", "year"), model = "within")`
- **Python:** `linearmodels.PanelOLS.from_formula("inv ~ value + capital + EntityEffects", ...)`
- **Hayashi:** `xtset(df, firm, year)` then `fe(inv ~ value + capital, df)`

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | Within transformation should match to high precision |
| standard_errors | 1e-4 | Same tolerance as coefficients |

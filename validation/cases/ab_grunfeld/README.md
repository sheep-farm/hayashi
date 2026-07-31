# Arellano-Bond dynamic panel on Grunfeld investment

This validation case estimates an Arellano-Bond difference GMM model for dynamic panel investment demand.

## Status

`active`

## Model

```
inv_{it} = α inv_{i,t-1} + β_1 value_{it} + β_2 capital_{it} + η_i + ε_{it}
```

Estimated in first differences with `inv_{i,t-2}` as the (collapsed) instrument for `Δ inv_{i,t-1}`.

## Dataset

- **Name:** `wooldridge::grunfeld`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 200 observations (10 firms × 20 years).

## Reference implementation

- **R:** one-step difference GMM in base R (no `plm` dependency): `inv_{i,t-2}` instruments `Δ inv_{i,t-1}`; robust sandwich standard errors.
- **Python:** one-step difference GMM implemented directly with the same collapsed instrument and weight-matrix convention as the R reference and Hayashi/Greeners.
- **Hayashi:** `ab(inv ~ value + capital, df, id=firm, time=year, lags=1)`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | One-step GMM should match closely |
| standard_errors | 1e-3 | Same tolerance as coefficients |

# Arellano-Bond dynamic panel on Grunfeld investment

This validation case estimates an Arellano-Bond difference GMM model for dynamic panel investment demand.

## Model

```
inv_{it} = α inv_{i,t-1} + β_1 value_{it} + β_2 capital_{it} + η_i + ε_{it}
```

## Dataset

- **Name:** `wooldridge::grunfeld`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 220 observations (10 firms × 22 years).

## Reference implementation

- **R:** `plm::pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:3), data = grunfeld, effect = "individual", model = "onestep", transformation = "d")`
- **Hayashi:** `ab(inv ~ value + capital, df, id=firm, time=year, lags=1)`
- **Python:** not used; no standard Arellano-Bond implementation in Python matches the same syntax.

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | One-step GMM should match closely |
| standard_errors | 1e-3 | Same tolerance as coefficients |

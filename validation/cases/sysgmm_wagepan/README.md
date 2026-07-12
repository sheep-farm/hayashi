# System GMM (Blundell-Bond) on Wooldridge `wagepan`

This validation case estimates a system GMM dynamic panel model of log wages.

## Model

```
sysgmm(lwage ~ exper + expersq + married + union, df,
       id=nr, time=year, lags=2, step=2)
```

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **Python:** Two-step System GMM (Blundell-Bond) with lags 2 and 3 of `lwage` as instruments for the first-difference equations and lagged first differences of `y` and `X` as instruments for the level equations.
- **R:** `plm::pgmm(... | lag(lwage, 2:3), ..., model = "twosteps", transformation = "ld")` is kept as a cross-check but is not the active reference because `plm::pgmm` uses a broader lag structure and a slightly different weight matrix by default.
- **Hayashi:** `sysgmm(...)`

## Status

Pass — Hayashi matches the Python reference for System GMM coefficients and standard errors.

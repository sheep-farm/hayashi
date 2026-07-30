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
- **R:** Explicit base-R implementation of the same sorted-panel, stacked first-difference and level-equation System GMM contract used by Hayashi/Greeners.
- **Hayashi:** `sysgmm(...)`

`plm::pgmm` is not used as the active R oracle for this case because its
formula interface and default weighting conventions estimate a nearby, not
identical, System GMM variant.

## Status

Pass — Hayashi matches the R and Python references for System GMM coefficients and standard errors.

# System GMM (Blundell-Bond) on Wooldridge `wagepan`

This validation case estimates a system GMM dynamic panel model of log wages.

## Model

```
sysgmm(lwage ~ lwage_lag + exper + expersq + married + union, df,
       id=nr, time=year, lags=2)
```

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **R:** `plm::pgmm(lwage ~ lag(lwage, 1) + ... | lag(lwage, 2:99), ..., model = "twosteps", transformation = "ld")`
- **Hayashi:** `sysgmm(...)`

## Status

Blocked — Hayashi `sysgmm()` raises a singular-matrix error on this specification.
See [sheep-farm/hayashi#67](https://github.com/sheep-farm/hayashi/issues/67).

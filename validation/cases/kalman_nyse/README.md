# Local-level Kalman filter on NYSE returns

This validation case would compare estimated observation and state variances from a local-level Kalman filter.

## Model

```
kalman(df, return, model="ll")
```

## Dataset

- **Name:** `wooldridge::nyse`
- **Source:** R package `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **R:** `dlm::dlmMLE(...)` with `dlmModPoly(order = 1)`
- **Hayashi:** `kalman(df, return, model="ll")`

## Status

Pass — Hayashi `kalman()` now estimates `sigma_obs` and `sigma_state` by maximum likelihood and returns a printable result object. sigma_state is very small and the likelihood is flat in that direction, so the declared tolerance is 1e-3.

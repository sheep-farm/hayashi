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

Blocked — Hayashi `kalman()` adds filtered/smoothed columns to the DataFrame and returns `nil`, so the validation harness cannot capture the estimated variances.
See [sheep-farm/hayashi#65](https://github.com/sheep-farm/hayashi/issues/65).

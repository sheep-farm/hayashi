# Seemingly Unrelated Regressions on Wooldridge `grunfeld`

This validation case estimates a two-equation SUR (Zellner FGLS) on the Grunfeld investment data.

## Model

Two equations estimated jointly:

```
value ~ inv + capital
inv ~ value + capital
```

The Wooldridge `grunfeld` data names the investment variable `inv`.

## Dataset

- **Name:** `wooldridge::grunfeld`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 200 observations (10 firms × 20 years).

## Reference implementation

- **Python:** reference implementation using a manual Zellner FGLS estimator (NumPy/Pandas).
- **R:** reference implementation using `systemfit::systemfit`. Both references are run by the validation runner.
- **Hayashi:** `sur(df, value ~ inv + capital, inv ~ value + capital)`

## Compared quantities

- coefficients only
- keys are formatted as `{equation}:{variable}` (e.g. `value:inv`)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-1 | FGLS estimates and covariance approximations may differ slightly across implementations |

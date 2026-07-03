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

- **Python:** primary reference implementation (manual Zellner FGLS using OLS residuals and block-diagonal GLS).
- **R:** a reference script is provided but not currently exercised by the validation runner because the `systemfit` package is not installed. The validation currently relies on the Python reference.
- **Hayashi:** `sur(df, value ~ inv + capital, inv ~ value + capital)`

## Compared quantities

- coefficients only
- keys are formatted as `{equation}:{variable}` (e.g. `value:inv`)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-1 | FGLS estimates and covariance approximations may differ slightly across implementations |

# Unobserved Components Model on a simulated series

Validates `ucm(df, y, level="local_linear", seasonal="stochastic", period=12)`
against `statsmodels.tsa.statespace.UnobservedComponents` with a local-linear
level and deterministic seasonal (period=12).

## Data

`data/gen.py` creates a 120-observation monthly series with a linear trend,
a 12-period sinusoidal seasonal component, and small Gaussian noise.

## Comparison

- `sigma2.irregular`: estimated observation variance.
- `level_first` / `level_last`: first and last smoothed level states.

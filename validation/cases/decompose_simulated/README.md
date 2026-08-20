# Classical seasonal decomposition on a simulated series

This case validates the `decompose()` command in Hayashi against
`statsmodels.tsa.seasonal.seasonal_decompose`.

## Data

`data/gen.py` creates a 120-observation monthly series with a linear trend,
a 12-period sinusoidal seasonal component, and small Gaussian noise.

## Comparison

Because classical moving-average decomposition leaves the first and last
`period/2` observations undefined, we compare selected interior points
(observations 6 and 113, 0-indexed) of the trend, seasonal, and residual
components.

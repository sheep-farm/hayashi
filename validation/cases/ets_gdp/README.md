# ETS on US real GDP

This validation case estimates an exponential smoothing state-space model on US real GDP.

## Status

`blocked` — Greeners now prints a coefficient table, but Hayashi's `ets` uses a grid search over SSE while R/Python use MLE. The estimated smoothing parameters therefore differ beyond tolerance.

## Model

```
gdp_t = level + trend + seasonal + ε_t
```

with automatic model selection.

## Dataset

- **Name:** `statsmodels::macrodata`
- **Source:** `statsmodels.datasets` (Python) and Rdatasets mirror (R).
- **Licence:** Public teaching dataset.
- **Size:** 203 quarterly observations (1959Q1–2009Q3).

## Reference implementation

- **R:** `forecast::ets(gdp)`
- **Python:** `statsmodels.tsa.exponential_smoothing.ets.ETSModel(gdp).fit()`
- **Hayashi:** `ets(df, gdp)`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-2 | ETS parameterisation differs slightly across packages |
| standard_errors | 1e-2 | R fallback does not report analytical SEs |

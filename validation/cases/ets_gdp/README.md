# ETS on US real GDP

This validation case estimates an exponential smoothing state-space model on US real GDP.

## Status

`blocked` — Hayashi's `ets` prints the smoothing parameters (alpha, beta, gamma) as summary lines, not in a coefficient table with standard errors. The orchestrator cannot parse this output into the coefficient/standard-error structure used for comparison.

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

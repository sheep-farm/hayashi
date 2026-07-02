# VAR(2) on US real GDP and consumption

This validation case estimates a vector autoregression of order 2 on US real GDP and consumption.

## Status

`blocked` — Hayashi's `var` output reports only the residual covariance matrix (Sigma_u), not per-equation coefficients. The validation orchestrator therefore cannot compare the output against the per-equation coefficients from R and Python.

## Model

```
Y_t = c + A1 Y_{t-1} + A2 Y_{t-2} + ε_t
```

where `Y_t = (gdp_t, cons_t)`.

## Dataset

- **Name:** `statsmodels::macrodata`
- **Source:** `statsmodels.datasets` (Python) and Rdatasets mirror (R).
- **Licence:** Public teaching dataset.
- **Size:** 203 quarterly observations (1959Q1–2009Q3).

## Reference implementation

- **R:** `vars::VAR(macro[, c("gdp", "cons")], p = 2, type = "const")`
- **Python:** `statsmodels.tsa.api.VAR(macro[["gdp", "cons"]]).fit(maxlags=2)`
- **Hayashi:** `var(df, gdp, cons, lags=2)`

## Known differences

| Source | Output |
|---|---|
| Hayashi | Residual covariance matrix Sigma_u |
| R/Python | Per-equation coefficients and standard errors |

## Compared quantities

- coefficients
- standard errors

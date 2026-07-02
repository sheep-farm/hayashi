# ARIMA(1,1,1) on log US real GDP

This validation case estimates an ARIMA(1,1,1) model on the log of US real GDP.

## Status

`blocked` — Hayashi uses Hannan-Rissanen estimation, while the reference implementations use MLE. The AR and MA coefficients differ substantially, so the case cannot be validated until the estimator method is configurable or a Hannan-Rissanen reference is added.

## Model

```
Δ log(gdp_t) = c + φ Δ log(gdp_{t-1}) + θ ε_{t-1} + ε_t
```

## Dataset

- **Name:** `statsmodels::macrodata`
- **Source:** `statsmodels.datasets` (Python) and Rdatasets mirror (R).
- **Licence:** Public teaching dataset.
- **Size:** 203 quarterly observations (1959Q1–2009Q3).

## Reference implementation

- **R:** `forecast::Arima(log(gdp), order = c(1, 1, 1))`
- **Python:** `statsmodels.tsa.arima.model.ARIMA(log(gdp), order=(1, 1, 1)).fit()`
- **Hayashi:** `arima(df, lgdp, p=1, d=1, q=1)`

## Known differences

| Quantity | Hayashi | R/Python |
|---|---|---|
| ar.L1 | 0.567 | 0.934 |
| ma.L1 | -0.273 | -0.590 |
| sigma2 | not reported | reported |

## Compared quantities

- coefficients
- standard errors

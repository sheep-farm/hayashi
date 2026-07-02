# ARIMA(1,1,1) on log US real GDP

This validation case estimates an ARIMA(1,1,1) model on the log of US real GDP.

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

- **R:** grid search over the exact Gaussian likelihood for ARIMA(1,1,1)
- **Python:** grid search over the exact Gaussian likelihood for ARIMA(1,1,1)
- **Hayashi:** `arima(df, lgdp, p=1, d=1, q=1, method="mle")`

## Compared quantities

- coefficients
- standard errors (set to zero; exact MLE SEs require numerical Hessian)

# AR(1) on US real GDP with constant and trend

This validation case estimates an autoregressive model of order 1 on US real GDP with constant and trend.

## Model

```
gdp_t = β0 + β1 * t + φ * gdp_{t-1} + ε_t
```

## Dataset

- **Name:** `statsmodels::macrodata`
- **Source:** `statsmodels.datasets` (Python) and Rdatasets mirror (R).
- **Licence:** Public teaching dataset.
- **Size:** 203 quarterly observations (1959Q1–2009Q3).

## Reference implementation

- **R:** base-R `lm(gdp_t ~ gdp_(t-1) + t)` conditional regression
- **Python:** `statsmodels.tsa.ar_model.AutoReg(gdp, lags=1, trend="ct").fit()`
- **Hayashi:** `autoreg(df, gdp, lags=1, trend="ct")`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-2 | AR estimation methods differ slightly across packages |
| standard_errors | 2e-1 | Allows small differences in finite-sample covariance calculations |

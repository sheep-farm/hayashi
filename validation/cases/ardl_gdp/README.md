# ARDL(1,1) on US real GDP and consumption

This validation case estimates an autoregressive distributed lag model of order (1,1) on US real GDP and consumption.

## Model

```
gdp_t = c + α gdp_{t-1} + β_0 cons_t + β_1 cons_{t-1} + ε_t
```

## Dataset

- **Name:** `statsmodels::macrodata`
- **Source:** `statsmodels.datasets` (Python) and Rdatasets mirror (R).
- **Licence:** Public teaching dataset.
- **Size:** 203 quarterly observations (1959Q1–2009Q3).

## Reference implementation

- **R:** `lm(gdp_t ~ gdp_{t-1} + cons_t + cons_{t-1})`
- **Python:** `statsmodels.tsa.ar_model.AutoReg(gdp, lags=1, exog=pd.concat([cons, cons.shift(1)], axis=1))`
- **Hayashi:** `ardl(gdp ~ cons, df, lags=1, xlags=1)`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 2e0 | Intercept differs slightly due to sample alignment |
| standard_errors | 2e0 | Same tolerance as coefficients |

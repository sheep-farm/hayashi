# ETS on US real GDP

This validation case estimates a simple exponential smoothing (SES, ETS(A,N,N)) model on US real GDP.

## Status

`active`

## Model

```
ℓ_t = α y_t + (1 - α) ℓ_{t-1}
ŷ_{t+1} = ℓ_t
```

with α estimated by one-step SSE minimisation.

## Dataset

- **Name:** `statsmodels::macrodata`
- **Source:** `statsmodels.datasets` (Python) and Rdatasets mirror (R).
- **Licence:** Public teaching dataset.
- **Size:** 203 quarterly observations (1959Q1–2009Q3).

## Reference implementation

- **R:** SSE grid search over α in `[0.001, 0.999]` with first observation as initial level.
- **Python:** `statsmodels.tsa.holtwinters.SimpleExpSmoothing(gdp, initialization_method="estimated").fit(optimized=True)`
- **Hayashi:** `ses(df, gdp)`

## Compared quantities

- `alpha` (smoothing parameter)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-2 | SES estimates differ only at the boundary (α ≈ 1) |

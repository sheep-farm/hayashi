# Structural VAR on US real GDP and consumption

This validation case estimates a Cholesky-identified SVAR(2) on log US real GDP and consumption.

## Model

```
svar(df, gdp, cons, lags=2, id=cholesky)
```

## Dataset

- **Name:** `statsmodels::macrodata`
- **Source:** statsmodels / Rdatasets.
- **Licence:** Public domain.
- **Variables:** `gdp` (real GDP), `cons` (real consumption).

## Reference implementation

- **R:** `vars::VAR(..., p=2, type="const")`, then Cholesky of residual covariance divided by `T - (1 + k*p)`.
- **Python:** `statsmodels.tsa.VAR(...).fit(maxlags=2, trend="c")`, then Cholesky of residual covariance divided by `T - (1 + k*p)`.
- **Hayashi:** `svar(df, gdp, cons, lags=2, id=cholesky)`

## Compared quantities

- A matrix (identity) and B matrix (lower-triangular Cholesky factor).

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| a_matrix | 1e-6 | Identity matrix |
| b_matrix | 5e-2 | Residual covariance divisor differences |

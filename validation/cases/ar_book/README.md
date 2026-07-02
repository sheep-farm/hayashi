# AR(1) on simulated data from the Hayashi book

This validation case estimates an AR(1) model on the simulated DGP from Chapter 26 of the book.

## Status

`active`

## Model

```
y_t = 0.7 y_{t-1} + ε_t,  ε_t ~ N(0,1)
```

## Dataset

- **Name:** `simulated_ar1`
- **Source:** DGP from `book_pt_BR/codes/26_arma.hay`
- **Size:** 500 observations

## Reference implementation

- **R:** two-step Hannan-Rissanen linear approximation (long-AR proxy residuals, then OLS on lagged y).
- **Python:** two-step Hannan-Rissanen linear approximation.
- **Hayashi:** `arima(df, y, p=1, d=0, q=0)` (default Hannan-Rissanen).

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 5e-3 | AR estimates should match closely |
| standard_errors | 1e-2 | Slightly looser due to two-step approximation |

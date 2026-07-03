# VAR(1) on simulated data from the Hayashi book

This validation case estimates a bivariate VAR(1) model on the simulated DGP from Chapter 28 of the book.

## Status

`active`

## Model

```
y1_t = 0.5 y1_{t-1} + 0.2 y2_{t-1} + ε1_t
y2_t = 0.1 y1_{t-1} + 0.6 y2_{t-1} + ε2_t
```

## Dataset

- **Name:** `simulated_var1`
- **Source:** DGP from `book_pt_BR/codes/28_var.hay`
- **Size:** 300 observations

## Reference implementation

- **R:** equation-by-equation OLS with one lag.
- **Python:** equation-by-equation OLS with one lag.
- **Hayashi:** `var(df, y1, y2, lags=1)`.

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 5e-3 | VAR OLS estimates should match closely |
| standard_errors | 1e-2 | Small differences in degrees-of-freedom correction |

# GARCH(1,1) on simulated data from the Hayashi book

This validation case estimates a GARCH(1,1) model on the simulated DGP from Chapter 30 of the book.

## Status

`active`

## Model

```
e_t = σ_t z_t,  z_t ~ N(0,1)
σ_t² = 0.3 + 0.5 e_{t-1}²
```

## Dataset

- **Name:** `simulated_garch11`
- **Source:** DGP from `book_pt_BR/codes/30_garch.hay`
- **Size:** 500 observations

## Reference implementation

- **R:** `rugarch::ugarchfit` with sGARCH(1,1) and normal distribution.
- **Python:** `arch.arch_model` with GARCH(1,1) and normal distribution.
- **Hayashi:** `garch(df, e, p=1, q=1)`.

## Compared quantities

- coefficients (standard errors are not compared because GARCH implementations use different Hessian approximations)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 5e-2 | GARCH MLE estimates may differ slightly across optimizers |

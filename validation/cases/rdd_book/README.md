# Sharp regression discontinuity on simulated book DGP

This validation case estimates the treatment effect at a known cutoff using a
sharp regression discontinuity design.

## Dataset

- **Name:** `rdd_book`
- **Source:** Simulated DGP from Hayashi book Chapter 24.
- **Licence:** Simulated.
- **Size:** 1000 observations.
- **DGP:**
  ```
  x ~ Uniform(-1, 1)
  D = (x >= 0)
  y = 1.0 + 0.5*x + 2.0*D + N(0,1)
  ```

## Model

Hayashi call:

```
rd(y ~ x, 0.0, df)
```

## Reference implementation

- **Python:** primary reference implementation (local linear regression with triangular kernel and Imbens-Kalyanaraman bandwidth selector; HC1 standard errors).
- **R:** a reference script is provided and implements the local linear regression manually in base R (triangular kernel, Imbens-Kalyanaraman bandwidth selector, HC1 standard errors). It does not require the `rdrobust` package. When R is unavailable, the validation relies on the Python reference.

## Compared quantities

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients (tau) | 1e-1 | Point estimates should be close with identical bandwidth. |
| standard_errors (tau) | 1e-1 | HC1 variance estimator should match Hayashi. |

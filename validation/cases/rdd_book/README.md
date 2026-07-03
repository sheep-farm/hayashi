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

- **Python:** reference implementation using a local linear regression in NumPy/Pandas (triangular kernel and Imbens-Kalyanaraman bandwidth selector; HC1 standard errors).
- **R:** reference implementation using a local linear regression in base R (triangular kernel, Imbens-Kalyanaraman bandwidth selector, HC1 standard errors). Both references are run by the validation runner.

## Compared quantities

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients (tau) | 1e-1 | Point estimates should be close with identical bandwidth. |
| standard_errors (tau) | 1e-1 | HC1 variance estimator should match Hayashi. |

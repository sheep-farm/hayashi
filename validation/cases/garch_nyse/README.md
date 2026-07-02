# GARCH(1,1) on NYSE returns

This validation case estimates a GARCH(1,1) model for NYSE returns.

## Model

```
r_t = μ + ε_t
ε_t = σ_t z_t
σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2
```

## Dataset

- **Name:** `wooldridge::nyse`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 691 observations × 8 variables.

## Reference implementation

- **R:** `rugarch::ugarchfit(ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), ...), data = return)`
- **Python:** `arch.arch_model(return, vol="Garch", p=1, q=1).fit()`
- **Hayashi:** `garch(df, return, p=1, q=1)`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-2 | GARCH optimization methods differ slightly across packages |
| standard_errors | 1e-2 | Same tolerance as coefficients |

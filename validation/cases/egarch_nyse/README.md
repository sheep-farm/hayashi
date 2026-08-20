# EGARCH(1,1) on NYSE returns

This validation case estimates an EGARCH(1,1) model for NYSE returns.

## Model

```
r_t = μ + ε_t
log(σ_t^2) = ω + α |ε_{t-1}/σ_{t-1}| + γ (ε_{t-1}/σ_{t-1}) + β log(σ_{t-1}^2)
```

## Dataset

- **Name:** `wooldridge::nyse`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **R:** `rugarch::ugarchfit(ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1, 1)), ...), data = return)`
- **Python:** `arch.arch_model(return, vol="EGarch", p=1, q=1).fit()`
- **Hayashi:** `egarch(df, return, p=1, q=1)`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 5e-1 | EGARCH optimization methods differ across packages |
| standard_errors | 5e0 | Same rationale as coefficients |

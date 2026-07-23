# Panel-corrected standard errors on Wooldridge `wagepan`

This validation case estimates a pooled panel model with panel-corrected standard errors.

## Model

```
pcse(lwage ~ educ + exper + expersq + married + union, df, id=nr, time=year)
```

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **Python:** manual implementation of the Hayashi/Greeners PCSE convention.
- **R:** `plm::plm(..., model = "pooling")` followed by
  `plm::vcovBK(..., cluster = "time", type = "HC0")`.
- **Hayashi:** `pcse(lwage ~ educ + exper + expersq + married + union, df, id=nr, time=year)`

The Python reference computes the same covariance directly:

```text
beta = (X'X)^-1 X'y
sigma_ij = e_i'e_j / T
meat = sum_i sum_j sigma_ij X_i'X_j
V = (X'X)^-1 meat (X'X)^-1
```

Bare `plm::vcovBK(model)` uses defaults that differ from this validation
target on this case. The R reference therefore sets `cluster = "time"` and
`type = "HC0"` explicitly.

## Compared quantities

- Regression coefficients and panel-corrected standard errors.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | Hayashi PCSE text output displays coefficients to four decimals |
| standard_errors | 1e-4 | Hayashi PCSE text output displays standard errors to four decimals |

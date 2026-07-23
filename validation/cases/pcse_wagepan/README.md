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
- **R cross-check:** `plm::plm(..., model = "pooling")` followed by
  `plm::vcovBK(...)` is retained in `reference/run.R`, but is not an active
  reference because it uses a different packaged PCSE convention on this case.
- **Hayashi:** `pcse(lwage ~ educ + exper + expersq + married + union, df, id=nr, time=year)`

The active Python reference computes:

```text
beta = (X'X)^-1 X'y
sigma_ij = e_i'e_j / T
meat = sum_i sum_j sigma_ij X_i'X_j
V = (X'X)^-1 meat (X'X)^-1
```

## Compared quantities

- Regression coefficients and panel-corrected standard errors.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | Hayashi PCSE text output displays coefficients to four decimals |
| standard_errors | 1e-4 | Hayashi PCSE text output displays standard errors to four decimals |

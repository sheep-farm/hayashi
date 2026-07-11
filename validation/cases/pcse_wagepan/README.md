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

- **R:** `plm::plm(..., model = "pooling")` followed by `plm::vcovBK(...)`
- **Hayashi:** `pcse(lwage ~ educ + exper + expersq + married + union, df, id=nr, time=year)`

## Compared quantities

- Regression coefficients and panel-corrected standard errors.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | Pooled OLS coefficients identical |
| standard_errors | 1e-1 | PCSE formula differences across implementations |

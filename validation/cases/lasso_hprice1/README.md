# Lasso regression on Wooldridge `hprice1`

This validation case estimates a Lasso regression of house price on lot size, square footage and number of bedrooms.

## Status

`blocked` — Hayashi's `lasso` prints the coefficient table to stdout but returns `Nil`, so `export(..., "txt", ...)` cannot be used for automated comparison. The estimator would need to return a proper result object for this case to become passable.

## Model

```
price ~ lotsize + sqrft + bdrms
```

## Dataset

- **Name:** `wooldridge::hprice1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 88 observations × 11 variables.

## Reference implementation

- **R:** `glmnet::glmnet(scale(X), y, alpha = 1, lambda = 100.0)`
- **Python:** `sklearn.linear_model.Lasso(alpha=100.0).fit()`
- **Hayashi:** `lasso(price ~ lotsize + sqrft + bdrms, df)`

## Compared quantities

- coefficients
- standard errors (Lasso has no analytical SEs; references report zeros/NA)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-2 | Different solvers may converge to slightly different values |
| standard_errors | 1e-2 | Lasso standard errors are not meaningful; kept for structure |

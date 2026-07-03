# Lasso regression on Wooldridge `hprice1`

This validation case estimates a Lasso regression of house price on lot size, square footage and number of bedrooms.

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

- **R:** `glmnet::glmnet(X, y, alpha = 1, lambda = 1.0, standardize = TRUE)`
- **Python:** `sklearn.linear_model.Lasso(alpha=1.0, max_iter=10000, tol=1e-6).fit(X, y)`
- **Hayashi:** `lasso(price ~ lotsize + sqrft + bdrms, df)`

## Compared quantities

- coefficients
- standard errors (Lasso has no analytical SEs; references report zeros/NA)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 2e0 | Different solvers and internal standardisation produce small differences |
| standard_errors | 2e0 | Lasso standard errors are not meaningful; kept for structure |

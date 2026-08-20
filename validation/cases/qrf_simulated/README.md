# qrf_simulated

Quantile Regression Forest validation on simulated heteroskedastic data.

- DGP: `y = 3*x1 + N(0, 0.1*(1 + 0.5*x1))`, with `x2` irrelevant.
- Hayashi: `qrf(y ~ x1 + x2, df, quantiles="0.75", trees=50, depth=5)`.
- Reference: `quantile_forest.RandomForestQuantileRegressor` with matching hyperparameters.
- Output: OOB R^2 for the 0.75 quantile as a `variable,coef,std_err` CSV.

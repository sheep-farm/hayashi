# qrf_inf_simulated

Quantile Regression Forest inference validation on simulated heteroskedastic data.

- DGP: `y = 3*x1 + N(0, 0.1*(1+0.5*x1))`, with `x2` irrelevant.
- Hayashi: `qrf_inf(y ~ x1 + x2, df, q="0.75", boot=50, trees=50, depth=5)`.
- References: `grf::quantile_forest` (R) and `quantile_forest.RandomForestQuantileRegressor` (Python).
- Output: `variable,coef,std_err` CSV with the OOB R-squared at the 0.75 quantile.

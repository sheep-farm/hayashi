# xgboost_simulated

XGBoost regression validation on simulated data.

- DGP: `y = 3*x1 + N(0, 0.1)`, with `x2` irrelevant.
- Hayashi: `xgboost(y ~ x1 + x2, df, trees=50, lr=0.1, depth=3)`.
- Reference: `xgboost.XGBRegressor` with matching hyperparameters and fixed random state.
- Output: MSE and in-sample R^2 as a `variable,coef,std_err` CSV.

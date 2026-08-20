# gbm_simulated

Gradient Boosting regression validation on simulated data.

- DGP: `y = 3*x1 + N(0, 0.1)`, with `x2` irrelevant.
- Hayashi: `gbm(y ~ x1 + x2, df, trees=50, lr=0.1, depth=3)`.
- Reference: `sklearn.ensemble.GradientBoostingRegressor` with matching hyperparameters.
- Output: MSE and in-sample R^2 as a `variable,coef,std_err` CSV.

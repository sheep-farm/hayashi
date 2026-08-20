# bart_simulated

Bayesian Additive Regression Trees validation on simulated data.

- DGP: `y = 3*x1 + N(0, 0.1)`, with `x2` irrelevant.
- Hayashi: `bart(y ~ x1 + x2, df, trees=20, depth=3, iter=500, burnin=200)` (500 post-burn draws, 200 burn-in).
- Reference: `sklearn.ensemble.GradientBoostingRegressor` with the same number and depth of trees as a light approximation of the BART posterior mean.
- Output: MSE and posterior mean R^2 as a `variable,coef,std_err` CSV.

# bart_simulated

Bayesian Additive Regression Trees validation on simulated data.

- DGP: `y = 3*x1 + N(0, 0.1)`, with `x2` irrelevant.
- Hayashi: `bart(y ~ x1 + x2, df, trees=20, depth=3, iter=500, burnin=200)` (500 post-burn draws, 200 burn-in).
- References: R `gbm` and Python `sklearn.ensemble.GradientBoostingRegressor`, both boosting algorithms rather than BART.
- Output: MSE and posterior mean R^2 as a `variable,coef,std_err` CSV.

## Reference evidence

For both references, `coefficients.mse` and `coefficients.r_squared` are
`behavioural-proxy` comparisons. Matching predictive fit, tree count or depth
does not establish a BART posterior mean, posterior uncertainty or the same
fitting contract. The existing tolerances (MSE `0.5`, R-squared `0.05`) and
recorded status remain unchanged; a pass is predictive diagnostic agreement.
See [evidence classes](../../README.md#reference-evidence-classes).

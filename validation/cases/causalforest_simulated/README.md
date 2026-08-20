# causalforest_simulated

Causal Forest validation on simulated data.

- DGP: `y = 1 + 2*x1 - x2 + 0.5*treated + N(0,1)`, with `x1` and `x2` as covariates.
- Hayashi: `causalforest(y ~ treated, df, x="x1,x2", trees=200, depth=4)`.
- References: `grf::causal_forest` (R) and `econml.grf.CausalForest` (Python).
- Output: `variable,coef,std_err` CSV with the average treatment effect.

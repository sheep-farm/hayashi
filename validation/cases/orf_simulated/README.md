# orf_simulated

Orthogonal Random Forest validation on simulated data.

- DGP: `y = 1 + 2*x1 - x2 + 0.5*treated + 0.3*w1 - 0.2*w2 + N(0,1)`, with `x1` and `x2` as features and `w1` and `w2` as confounders.
- Hayashi: `orf(y ~ treated, df, x="x1,x2", w="w1,w2", trees=50, depth=4)`.
- References: `grf::causal_forest` (R) as an ATE proxy and `econml.orf.DROrthoForest` (Python).
- Output: `variable,coef,std_err` CSV with the average treatment effect.

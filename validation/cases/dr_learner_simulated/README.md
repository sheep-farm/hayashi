# dr_learner_simulated

Doubly-Robust Learner validation on simulated data.

- DGP: `y0 = 1 + 0.5*x + N(0, 0.5)`, constant ATE = 2.0, `d` drawn from a propensity score depending on `x`.
- Hayashi: `dr_learner(y ~ d + x, df, x="x")`.
- Reference: manual AIPW with logistic propensity and a gradient boosting outcome model.
- Output: ATE and its standard error as a `variable,coef,std_err` CSV.

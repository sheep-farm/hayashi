# dml_crossfit_simulated

Double Machine Learning with cross-fitting validation.

- DGP: `y = 1.5*d + 0.5*x1 + 0.3*x2 + N(0, 0.25)` with binary `d` driven by `x1` and `x2`.
- Hayashi: `dml_crossfit(y ~ dvar, df2, x="x1,x2", folds=5)` after recentering `dvar` to 0/1.
- R reference: manual 5-fold cross-fitting with linear nuisance models.
- Python reference: `sklearn` 5-fold cross-fitting with `LinearRegression`.
- Output: partial-linear causal coefficient `theta` and its SE.

# bayes_lm_simulated

Bayesian linear regression validation on simulated data.

- DGP: `y = 1 + 2x1 - 1.5x2 + N(0, 0.25)` with 200 observations.
- Hayashi: `bayes_lm(y ~ x1 + x2, df)`.
- R reference: `lm(y ~ x1 + x2)`.
- Python reference: `statsmodels.OLS`.
- Output: posterior means of `x1` and `x2`. With a diffuse Normal-Inverse-Gamma prior these should match OLS closely.

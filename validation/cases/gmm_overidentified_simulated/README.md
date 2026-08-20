# Overidentified two-step GMM

This case validates Hayashi's generic linear `gmm()` command on a deterministic
heteroskedastic IV data-generating process. It generates 1,000 observations
with NumPy seed `20260815`.

The structural equation has an intercept, one exogenous regressor (`x`), and
one endogenous regressor (`endog`). The latter shares a shock with the outcome
error. The two excluded instruments (`z1`, `z2`) are independent of that error,
giving four instruments for three estimated coefficients and one
overidentifying restriction.

All three implementations use two-step robust weighting based on first-step
residual moments. Greeners reports the corresponding efficient inverse-form
variance. R `gmm::gmm()` and Python `linearmodels.iv.IVGMM` report a
final-residual sandwich covariance. On this DGP, the maximum observed standard
error difference is `2.569e-6`, which is accepted within the declared `1e-5`
tolerance; coefficients and Hansen's J agree to numerical precision. The
runner also compares the sample size, J statistic, and overidentification
degrees of freedom against both references.

Run this case with:

```bash
python validation/run.py --case gmm_overidentified_simulated
```

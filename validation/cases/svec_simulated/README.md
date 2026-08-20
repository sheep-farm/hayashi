# SVEC — long-run restrictions

Validates Hayashi's `svec()` (Blanchard-Quah long-run identification) against a
Python reference that implements the same steps explicitly.

## Model

A stable bivariate VAR(1) with 250 observations.

## Reference

The Python reference:
1. Estimates the reduced-form VAR(1) by OLS.
2. Computes the residual covariance `Sigma_u`.
3. Computes `C(1) = (I - A1)^{-1}`.
4. Computes the long-run covariance `C(1) @ Sigma_u @ C(1)'`.
5. Takes the lower Cholesky factor.
6. Recovers `B = C(1)^{-1} @ long_run_chol` and `A = I`.

This matches the procedure used by Greeners.

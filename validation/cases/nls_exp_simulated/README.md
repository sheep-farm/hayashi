# NLS exponential model

Validates Hayashi's `nls_exp()` for the model `y = a * exp(b * x) + ε`.

## Data

Simulated with `a = 2.5`, `b = -0.8`, `x` in `[0.1, 2.0]`, and Gaussian noise.

## References

- **R**: `nls(y ~ a * exp(b * x), data=df, start=list(a=2.0, b=-1.0))`
- **Python**: `scipy.optimize.curve_fit` with the same model and Levenberg-Marquardt.

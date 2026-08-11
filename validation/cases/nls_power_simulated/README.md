# NLS power model

Validates Hayashi's `nls_power()` for the model `y = a * x^b + ε`.

## Data

Simulated with `a = 2.0`, `b = 0.8`, `x` in `[0.5, 5.0]`, and Gaussian noise.

## References

- **R**: `nls(y ~ a * I(x^b), data=df, start=list(a=1.5, b=0.5))`
- **Python**: `scipy.optimize.curve_fit` with the same power model.

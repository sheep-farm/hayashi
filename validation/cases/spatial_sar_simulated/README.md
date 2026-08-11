# Spatial autoregressive (SAR) model

Simulated 7×7 grid with rook contiguity weights `W`, row-standardised.

## Data generation

- `y = (I - 0.3 W)^{-1} (0.5 x + ε)`
- `x` and `ε` are standard normal.

## Reference

Python reference implements the concentrated MLE for SAR independently:
grid search over `rho`, then OLS for `beta` on `y - rho*W*y`.
This verifies the Hayashi/Greeners implementation is self-consistent;
a future improvement is to add an independent `spdep` R reference.

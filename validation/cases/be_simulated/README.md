# Between estimator — simulated panel

This case validates Hayashi's `be()` (between estimator) against an explicit
OLS-on-entity-means reference.

## Model

For each entity `i` and period `t`:

```
y_it = beta * x_it + alpha_i + e_it
```

`be()` collapses each entity to its within-entity means:

```
y_bar_i = beta * x_bar_i + (alpha_i + e_bar_i)
```

and runs OLS on the `N` collapsed observations.

## Reference

The Python reference:
1. Computes `y_bar_i` and `x_bar_i` by group.
2. Adds an intercept.
3. Runs OLS via `numpy.linalg.lstsq`.
4. Reports the `_cons` and `x` coefficients and homoskedastic standard errors.

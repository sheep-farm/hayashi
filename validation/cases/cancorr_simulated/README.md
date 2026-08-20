# Canonical correlation on simulated data

Validates `cancorr(df, xvars=["x1","x2"], yvars=["y1","y2"])` against the
generalised-eigenvalue formulation of canonical correlation analysis.

## Data

`data/gen.py` creates 200 observations with two correlated X variables and two
Y variables built from linear combinations of the Xs plus noise.

## Comparison

- `cancorr_1` and `cancorr_2`: the two canonical correlations.
- `wilks_lambda`: product of (1 - rho^2) over all canonical correlations.

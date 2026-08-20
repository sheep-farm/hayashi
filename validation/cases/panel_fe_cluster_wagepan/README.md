# panel_fe_cluster_wagepan

Validates panel fixed effects with entity-clustered standard errors on the
Wooldridge `wagepan` dataset.

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`; Python package `wooldridge` with an
  Rdatasets CSV fallback
- **Licence:** public teaching dataset
- **Size:** 4,360 observations across 545 workers

## Intended Analysis

The intended model is the Wooldridge Chapter 14 fixed-effects wage equation:

```text
lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87
```

with worker-level clustered standard errors:

```hayashi
xtset(df, nr, year)
fe(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87, df, cluster=nr)
```

## Reference Contract

The R and Python references implement the same calculation directly rather than
calling high-level panel packages with package-specific small-sample defaults:

1. Drop incomplete rows for `lwage`, regressors, `nr`, and `year`.
2. Demean `lwage` and each regressor by worker id `nr`.
3. Estimate OLS without an intercept on the demeaned design.
4. Compute one-way clustered covariance by worker id.
5. Apply the Greeners CR1 finite-sample correction:
   `(G / (G - 1)) * ((N - 1) / (N - K))`.

The case compares:

- coefficients
- entity-clustered standard errors

The tolerance is `1e-4` because Hayashi's plain-text coefficient table rounds
reported coefficients and standard errors to four decimal places.

# panel_fe_time_cluster_wagepan

Panel fixed-effects wage equation with time-clustered standard errors on the Wooldridge `wagepan` dataset.

## Model

```hayashi
xtset(df, nr, year)
let m = fe(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87, df, cluster=year)
```

## Reference

Both R and Python references implement within-transformed OLS with a one-way CR1 clustered covariance grouped by `year`. The finite-sample correction is `G/(G-1) * (N-1)/(N-K)`, where `G` is the number of time clusters.

## Compared quantities

- Coefficients
- Standard errors

Tolerances: `1e-4` for both.

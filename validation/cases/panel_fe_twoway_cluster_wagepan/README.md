# panel_fe_twoway_cluster_wagepan

Panel fixed-effects wage equation with two-way (entity + time) clustered standard errors on the Wooldridge `wagepan` dataset.

## Model

```hayashi
xtset(df, nr, year)
let m = fe(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87, df, cluster=nr, cluster2=year)
```

## Reference

Both R and Python references implement within-transformed OLS with two-way clustered covariance using the Cameron-Gelbach-Miller additive decomposition:

```
V = V_entity + V_time - V_intersection
```

The meat matrices are computed at the entity, time, and entity×time interaction levels. The small-sample correction uses `g = min(G_entity, G_time)` clusters: `g/(g-1) * (n-1)/(n-k)`. This matches the Greeners implementation.

## Compared quantities

- Coefficients
- Standard errors

Tolerances: `1e-4` for both.

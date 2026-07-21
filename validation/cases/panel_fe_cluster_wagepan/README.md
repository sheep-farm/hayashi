# panel_fe_cluster_wagepan

Tracks the currently unsupported validation target for panel fixed effects with
entity-clustered standard errors on the Wooldridge `wagepan` dataset.

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

## Current Status

This case is registered as `not-supported`.

The current `fe()` interpreter path resolves the panel entity id and calls
`FixedEffects::from_formula(...)`; it does not consume `cluster=...`.
Greeners' fixed-effects implementation then estimates the within model with
non-robust OLS covariance. Direct runtime comparison of `fe(..., df)` and
`fe(..., df, cluster=nr)` on this dataset produced identical standard errors.

See issue #93.

## Activation Criteria

Convert this case to `active` only after Hayashi implements clustered covariance
for `fe()`. The active case should compare:

- coefficients
- entity-clustered standard errors

Suggested references:

- **R:** `plm` fixed effects with entity-clustered covariance, or an explicit
  within-transformed cluster sandwich implementation.
- **Python:** `linearmodels.PanelOLS(...).fit(cov_type="clustered",
  cluster_entity=True)`, or an explicit within-transformed cluster sandwich
  implementation.

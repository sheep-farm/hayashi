# Bootstrap

## Basic usage

```
let b = bootstrap(ols, Y ~ X1 + X2, df, n = 1000)
```

Resamples `df` with replacement `n` times, estimates the model on each sample, and collects the coefficient distributions. Prints a summary table with bootstrap standard errors, bias, and percentile confidence intervals.

## Reproducibility

```
set_seed(42)
let b = bootstrap(ols, Y ~ X1 + X2, df, n = 1000)
```

`set_seed` fixes the random number generator so results are exactly reproducible across runs.

## Works with any estimator

`bootstrap` accepts any estimator function as its first argument:

```
bootstrap(logit, Y ~ X1 + X2, df, n = 500)
bootstrap(probit, Y ~ X1 + X2, df, n = 500)
bootstrap(iv, Y ~ X1 | Z1 + Z2, df, n = 1000)
bootstrap(fe, Y ~ X1, df, n = 1000)
bootstrap(re, Y ~ X1, df, n = 1000)
```

Estimator options pass through normally:

```
bootstrap(ols, Y ~ X1 + X2, df, n = 1000, cov = robust)
```

## Output

`bootstrap` returns a record with:

| Field | Description |
|---|---|
| `coefs` | n x k matrix of bootstrap coefficient draws |
| `se` | bootstrap standard errors |
| `ci_lower` | 2.5th percentile |
| `ci_upper` | 97.5th percentile |
| `bias` | mean(bootstrap) - original estimate |

Access individual fields:

```
let b = bootstrap(ols, Y ~ X1 + X2, df, n = 1000)
print(b.se)
print(b.ci_lower)
```

## Notes

- The default is nonparametric bootstrap (case resampling). This is valid for cross-sectional data.
- For panel data, `bootstrap` resamples at the group level (cluster bootstrap) automatically when used with `fe` or `re`.
- Computation scales linearly with `n`. For large datasets, start with `n = 200` to verify the setup, then increase.

# Diagnostics

## Variance inflation factors

```
vif(m)
```

Prints VIF for each regressor. A VIF above 10 signals problematic multicollinearity.

## Condition number

```
condnum(m)
```

Reports the condition number of the regressor matrix. Values above 30 suggest ill-conditioning.

## Model comparison — AIC / BIC

```
estat(m1, m2, m3)
```

Prints a table with N, log-likelihood, AIC, and BIC for each model, sorted by AIC.

## Influence diagnostics

```
influence(m)
```

Returns a DataFrame with one row per observation containing:

| Column | Description |
|---|---|
| `cooksd` | Cook's distance |
| `dffits` | DFFITS |
| `leverage` | Hat-matrix diagonal |

Use it for outlier and leverage analysis:

```
let inf = influence(m)
keep inf if cooksd > 4 / nrow(df)
print(inf)
```

## Coefficient plot

```
coefplot(m)
```

Draws an ASCII coefficient plot in the terminal, showing point estimates and 95% confidence intervals. Useful for quick visual inspection without leaving the REPL.

## Drop collinear variables

```
let df2 = drop_collinear(df)
```

Scans a DataFrame and removes perfectly collinear columns, returning a clean copy. Useful as a preprocessing step before estimation.

## Automatic collinearity detection

All estimators in Hayashi detect exact collinearity at estimation time. When a regressor is a perfect linear combination of others, it is silently dropped and a note appears in the estimation output. You do not need to run `drop_collinear` before estimating — it exists for cases where you want to inspect or control the process manually.

## Notes

- `vif` and `condnum` require an OLS or IV result.
- `estat` works with any estimator that reports a log-likelihood.
- `influence` is currently implemented for OLS only.

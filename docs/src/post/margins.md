# Margins & Predictions

## Average marginal effects

```
let m = logit(Y ~ X1 + X2, df)
margins(m, df)
```

Computes average marginal effects (AME) over the estimation sample. For linear models, AME equals the coefficients; for nonlinear models (logit, probit), it averages the partial effects across observations.

## Fitted values

```
predict df yhat = m
```

Adds a column `yhat` to `df` containing the predicted values from model `m`. For linear models this is X * beta; for binary models it is the predicted probability.

## Residuals

```
predict df e = m "residuals"
```

Adds a column `e` with OLS residuals (Y - Xb). The third argument selects what to predict:

| Keyword | Result |
|---|---|
| (none) | fitted values |
| `"residuals"` | residuals |
| `"stdresid"` | standardized residuals |

## Delta method — nonlinear combinations

```
let m = ols(Y ~ X1 + X2, df)
nlcom(m, X1 / X2)
```

Tests a nonlinear function of coefficients using the delta method. Reports the point estimate, standard error, and 95% CI.

## Linear combinations

```
lincom(m, X1 = 1, X2 = 1)
```

Tests the linear combination beta_X1 + beta_X2. Reports the combined estimate, standard error, t-statistic, and p-value.

More generally, `lincom(m, X1 = 2, X2 = -1)` tests 2 * beta_X1 - beta_X2 = 0.

## Notes

- `margins` returns a record with fields `effects`, `se`, and `pval` for each variable.
- `predict` modifies the DataFrame in place. The model and DataFrame need not share the same sample — out-of-sample prediction works as long as the required columns exist.
- `nlcom` accepts any expression involving coefficient names. Use parentheses for clarity: `nlcom(m, (X1 - X2) / X3)`.

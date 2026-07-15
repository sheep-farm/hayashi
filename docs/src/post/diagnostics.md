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

Prints a table with N, log-likelihood, AIC, and BIC for each model, sorted by AIC. When comparing multiple models, Akaike weights are also reported.

## Akaike weights

```
let w = akaike_weights(m1, m2, m3)
```

Returns a dict `{model_name: weight}` with Akaike weights for programmatic model comparison. Also prints a summary table with AIC, ΔAIC, and weights. Supports OLS, logit/probit, Poisson, NegBin, Tobit, Ordered, Mixed, and Zero-Inflated models.

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

## CUSUM test for structural stability

```
cusumtest(m)
```

CUSUM test (Brown, Durbin, Evans 1975) for parameter stability. Uses recursive residuals and checks if the cumulative sum stays within 5% significance bounds. Reports whether the model is stable and the maximum |CUSUM| statistic. Supports OLS models.

## ACF and PACF

```
let a = acf(df, returns, lags=20)
let p = pacf(df, returns, lags=20)
```

Returns autocorrelation / partial autocorrelation values as a list of length `lags + 1` (element 0 is always 1.0). Also accepts a model (OLS, GARCH, ARIMA) — uses residuals in that case.

For ASCII correlogram visualizations, use `acfplot(df, var, lags=20)` and `pacfplot(df, var, lags=20)` instead.

## Goldfeld-Quandt test

```
gqtest(m, split=0.2)
```

Tests for heteroskedasticity by comparing the variance of residuals in the first and last portions of the sample (split fraction defaults to 0.2). Reports an F-statistic and p-value.

## IV diagnostics

### Weak instrument test

```
weak_iv(y ~ x_endog, ~ z1 + z2, df)
```

Computes first-stage F-statistics for each endogenous variable (partial F on excluded instruments), the Cragg-Donald Wald F statistic, and Stock-Yogo (2005) critical values. Rule of thumb: F > 10 indicates strong instruments (Staiger & Stock 1997).

### Sargan / Hansen J overidentification test

```
estat_overid(y ~ x1 + x_endog, ~ z1 + z2, df)
```

Tests H0: the instruments are exogenous (valid). Only applicable when overidentified (more instruments than endogenous regressors). The Sargan statistic is n × R² from the regression of IV residuals on all instruments, distributed as chi²(L - K). If rejected, the instruments may be invalid.

### Durbin-Wu-Hausman endogeneity test

```
estat_endog(y ~ x1 + x_endog, ~ z1 + z2, df)
```

Tests H0: the endogenous regressors are actually exogenous (OLS is consistent). Uses the augmented regression approach: adds first-stage residuals to the structural equation and tests their significance via an F-test. If rejected, IV is needed. If not rejected, OLS is consistent and preferred (more efficient).

## Binary model diagnostics (logit / probit)

### Classification table

```
estat_classification(m_logit)
estat_classification(m_logit, threshold=0.3)
```

Computes the 2×2 classification table at the given threshold (default 0.5). Reports true positives, true negatives, false positives, false negatives, sensitivity (recall), specificity, and overall correct rate.

### ROC / AUC

```
lroc(m_logit)
```

Computes the area under the ROC curve (AUC) using the rank-based Wilcoxon-Mann-Whitney statistic. Also reports the Gini coefficient (2·AUC − 1). Interpretation: AUC ≥ 0.9 excellent, ≥ 0.8 good, ≥ 0.7 acceptable, ≥ 0.6 poor, < 0.6 no discrimination.

### Hosmer-Lemeshow goodness-of-fit

```
estat_gof(m_logit)
estat_gof(m_logit, groups=5)
```

Hosmer-Lemeshow goodness-of-fit test. Divides observations into `groups` deciles (default 10) based on predicted probabilities and compares observed vs expected counts via a chi-squared statistic. H0: the model fits adequately. If rejected (p < 0.05), the model does not fit well.

### Linktest (specification error detection)

```
linktest(m_logit)
```

Stata's `linktest` for specification error detection. Re-estimates the model using the linear predictor ŷ = Xβ and ŷ² as the only regressors. If the coefficient on ŷ² is statistically significant (p < 0.05), there is a specification error — either the link function is inadequate or the functional form is wrong. H0: the model is correctly specified (coefficient of ŷ² = 0).

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
- `estat` and `akaike_weights` work with any estimator that reports a log-likelihood.
- `influence` and `cusumtest` are currently implemented for OLS only.
- `gqtest` requires an OLS model.

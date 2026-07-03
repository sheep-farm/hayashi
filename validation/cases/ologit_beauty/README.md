# Ordered logit of beauty rating on Wooldridge `beauty`

This validation case estimates an ordered logit model of self-reported beauty
rating (`looks`) on the Wooldridge `beauty` dataset.

## Model

```
looks ~ female + educ + exper + black
```

## Dataset

- **Name:** `wooldridge::beauty`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 1,260 observations, subset to 1,228 rows where `looks` is 2, 3, or 4.

## Reference implementation

- **R:** `MASS::polr(factor(looks) ~ female + educ + exper + black, data = df, method = "logistic", Hess = TRUE)`
- **Python:** `statsmodels.miscmodels.ordinal_model.OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)`
- **Hayashi:** `ologit(looks ~ female + educ + exper + black, df)`

## Compared quantities

- Regression coefficients only: `female`, `educ`, `exper`, `black`.
- Standard errors (non-robust, from the inverse Hessian) for the same coefficients.
- Thresholds/cuts are excluded from the comparison.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | MLE optimisation may differ slightly across solvers |
| standard_errors | 5e-2 | Numerical inverse-Hessian approximations can diverge modestly |

# Logit average marginal effects on Wooldridge `mroz`

This validation case estimates the same labour-force-participation logit model
as `logit_mroz`, then compares `margins(m)` against independent average
marginal effects and delta-method standard errors.

## Model

```
inlf ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6
```

## Dataset

- **Name:** `wooldridge::mroz`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 753 observations x 22 variables.

## Reference implementation

- **R:** `glm(..., family = binomial(link = "logit"))`, with AMEs and
  delta-method standard errors computed from `coef(model)` and `vcov(model)`.
- **Python:** `statsmodels.logit(...).fit().get_margeff(at="overall",
  method="dydx")`.
- **Hayashi:** `logit(inlf ~ ..., df)` followed by `margins(m)`.

## Compared quantities

- average marginal effects
- delta-method standard errors

## Current status

Active and passing on current `dev`. Hayashi's average marginal effects and
delta-method standard errors match both R and Python within the declared
tolerance.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---:|---|
| marginal_effects | 1e-4 | Hayashi prints the margins table to six decimals |
| standard_errors | 1e-4 | Same displayed-precision constraint as AMEs |

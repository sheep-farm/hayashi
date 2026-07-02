# Probit labour-force participation on Wooldridge `mroz`

This validation case estimates a probit model of labour-force participation.

## Model

```
inlf ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6
```

## Dataset

- **Name:** `wooldridge::mroz`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 753 observations × 22 variables.

## Reference implementation

- **R:** `glm(inlf ~ ..., data = mroz, family = binomial(link = "probit"))`
- **Python:** `statsmodels.probit("inlf ~ ...", data = mroz)`
- **Hayashi:** `probit(inlf ~ ..., df)`

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | MLE optimisation may differ slightly across solvers |
| standard_errors | 1e-4 | Same tolerance as coefficients |

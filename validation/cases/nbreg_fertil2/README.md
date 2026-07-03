# Negative binomial regression on Wooldridge `fertil2`

This validation case estimates a Negative Binomial regression for the number of children on age, education, an electric indicator and an urban indicator using the Wooldridge `fertil2` dataset.

## Model

```
children ~ age + educ + electric + urban
```

## Dataset

- **Name:** `wooldridge::fertil2`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 4,361 observations × 27 variables.

## Reference implementation

- **R:** `MASS::glm.nb(children ~ age + educ + electric + urban, data = df)`
- **Python:** `statsmodels.formula.api.negativebinomial("children ~ age + educ + electric + urban", data=df).fit(disp=0)`
- **Hayashi:** `negbin(children ~ age + educ + electric + urban, df)`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 2e-1 | MLE optimisation and alpha estimation may differ slightly across solvers; the intercept differs by ~0.11 due to different dispersion estimates |
| standard_errors | 5e-1 | Numerical inverse-Hessian approximations and the estimated dispersion can diverge modestly |

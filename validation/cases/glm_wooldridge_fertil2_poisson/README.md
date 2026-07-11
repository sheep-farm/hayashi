# GLM Poisson children count on Wooldridge fertil2

This validation case estimates a Poisson GLM for the number of children on Wooldridge's *fertil2* dataset:

```
children ~ age + electric + educ + urban + tv
```

## Dataset

- **Name:** `wooldridge::fertil2`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 4,361 observations × 27 variables.

## Reference implementation

- **R:** `glm(children ~ age + electric + educ + urban + tv, data = fertil2, family = poisson(link = "log"))`
- **Python statsmodels:** `glm("children ~ age + electric + educ + urban + tv", data=fertil2, family=Poisson(link=Log())).fit()`
- **Stata:** `glm children age electric educ urban tv, family(poisson)` (optional)

## Compared quantities

- coefficients
- standard errors

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | MLE should match closely across implementations |
| standard_errors | 1e-4 | Same precision target |

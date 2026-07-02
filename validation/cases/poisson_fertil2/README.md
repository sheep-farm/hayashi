# Poisson fertility model on Wooldridge `fertil2`

This validation case estimates a Poisson regression for the number of children.

## Model

```
children ~ educ + age + agesq + evermarr + urban + electric + tv
```

## Dataset

- **Name:** `wooldridge::fertil2`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 4,361 observations × 27 variables.

## Reference implementation

- **R:** `glm(children ~ ..., data = fertil2, family = poisson)`
- **Python:** `statsmodels.glm("children ~ ...", family = Poisson())`
- **Hayashi:** `poisson(children ~ ..., df)`

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | MLE optimisation should converge to the same maximum |
| standard_errors | 1e-4 | Same tolerance as coefficients |

# Zero-inflated negative binomial on Wooldridge `affairs`

This validation case estimates a ZINB model for the number of affairs.

## Model

```
zinb(naffairs ~ age + yrsmarr + kids + educ + relig + ratemarr, df)
```

The inflation equation uses the same regressors.

## Dataset

- **Name:** `wooldridge::affairs`
- **Source:** R package `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **R:** `pscl::zeroinfl(naffairs ~ ..., data = affairs, dist = "negbin", link = "logit")`
- **Hayashi:** `zinb(naffairs ~ ..., df)`

## Compared quantities

- Count-model and inflation-model coefficients and standard errors.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 5e-2 | Zero-inflated EM/optimisation differences |
| standard_errors | 1e-1 | Hessian approximation differences |

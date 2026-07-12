# Beta regression on Wooldridge `401k`

This validation case estimates a beta regression of 401(k) participation rate on plan characteristics.

## Model

```
prate ~ mrate + age + sole
```

## Dataset

- **Name:** `wooldridge::k401k`
- **Source:** R package `wooldridge`.
- **Licence:** Public teaching dataset.
- **Preprocessing:** `prate` is scaled from percentage to proportion and bounded away from 0 and 1.

## Reference implementation

- **R:** `betareg::betareg(prate ~ mrate + age + sole, data = df)`
- **Hayashi:** `betareg(prate ~ mrate + age + sole, df)`

## Compared quantities

- Regression coefficients and standard errors for `const`, `mrate`, `age`, `sole`.

## Status

Pass — Greeners now estimates beta regression by BFGS with an analytic gradient and computes standard errors from the observed inverse Hessian, matching R betareg.

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | MLE optimisation differences |
| standard_errors | 1e-2 | Inverse-Hessian approximation differences |

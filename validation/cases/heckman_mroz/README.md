# Heckman two-step selection model on Wooldridge `mroz`

This validation case estimates a Heckman two-step (Heckit) selection model.

## Model

Outcome equation (observed only for women in the labour force):

```
lwage ~ educ + exper + expersq
```

Selection equation:

```
inlf ~ educ + age + kidslt6 + kidsge6 + nwifeinc
```

## Dataset

- **Name:** `wooldridge::mroz`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 753 observations; this case uses the subset `inlf`, `lwage`, `educ`, `age`, `kidslt6`, `kidsge6`, `nwifeinc`, `exper`, `expersq`.
- **Note:** `lwage` is missing for women not in the labour force. The stored CSV recodes those missing values to `0` so the column is loaded as numeric; only the selected observations (`inlf == 1`) enter the outcome equation.

## Reference implementation

- **Python:** reference implementation using a manual two-step estimator in NumPy/Pandas (`statsmodels.probit` or an internal Newton-Raphson probit, a manually computed inverse Mills ratio, and OLS with the Heckman (1979) corrected covariance matrix).
- **R:** reference implementation using a manual two-step estimator in base R (`glm(..., family = binomial(link = "probit"))`, a manually computed inverse Mills ratio, and OLS with the Heckman (1979) corrected covariance matrix). Both references are run by the validation runner.
- **Hayashi:** `heckman(lwage ~ educ + exper + expersq, inlf ~ educ + age + kidslt6 + kidsge6 + nwifeinc, df)`

## Compared quantities

- coefficients
- standard errors (two-step Heckman corrected SEs)

The comparison focuses on `educ`, `exper`, `expersq`, and `lambda_IMR` (the coefficient on the inverse Mills ratio).

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 5e-2 | Two-step Heckman estimates can differ slightly depending on the probit solver and finite-sample corrections |
| standard_errors | 1e-1 | Two-step standard errors are approximate and depend on the exact covariance correction formula |

## Notes

- The Python and R references do not use `sampleSelection::heckit` or `pyheckit`; they implement the two-step estimator manually in base R / Python so the case runs without installing extra packages.
- If the manual implementations diverge from Hayashi beyond the declared tolerances, the tolerances may be adjusted and the rationale documented here.

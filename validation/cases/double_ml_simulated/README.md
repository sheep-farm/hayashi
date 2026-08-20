# Double Machine Learning on simulated data

This validation case implements Double Machine Learning (Chernozhukov et al. 2018) on simulated data with known treatment effect.

## Dataset

- **Name:** `simulated_double_ml`
- **Source:** Simulated DGP following Chernozhukov et al. (2018)
- **Licence:** MIT
- **Size:** 1000 observations with 5 confounders

## Reference implementation

- **Python:** `DoubleML` package or custom implementation using sklearn

## Compared quantities

- ate_coefficient (average treatment effect)
- ate_standard_error (standard error of ATE)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| ate_coefficient | 1e-2 | ML models may have small differences in nuisance estimation |
| ate_standard_error | 1e-2 | Bootstrap SE may vary slightly |

## Notes

Double Machine Learning uses ML models to estimate nuisance functions (outcome model and propensity score) and then computes the treatment effect using orthogonalization. This case uses a simple DGP with a known treatment effect of 0.5. The reference implementation uses Random Forest for nuisance estimation.

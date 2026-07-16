# MICE chained equations on simulated data with missing values

This validation case implements MICE (Multiple Imputation by Chained Equations, van Buuren 2011) on simulated data with MCAR missing values.

## Dataset

- **Name:** `simulated_mice`
- **Source:** Simulated data with MCAR missing values
- **Licence:** MIT
- **Size:** 200 observations × 3 variables, 20% missing values

## Reference implementation

- **R:** `mice::mice()` with method="pmm" (predictive mean matching)
- **Python:** Custom implementation using sklearn iterative imputer

## Compared quantities

- imputed_means (mean of imputed values for each variable)
- imputed_stds (standard deviation of imputed values for each variable)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| imputed_means | 1e-3 | MICE is stochastic; small differences expected |
| imputed_stds | 1e-3 | MICE is stochastic; small differences expected |

## Notes

MICE uses chained equations to impute missing values iteratively. This case uses simulated data with MCAR (Missing Completely At Random) missing values. The reference implementations use the same number of imputations (m=5) and iterations (iter=10). Due to the stochastic nature of MICE, exact matches are not expected, but the means and standard deviations should be close.

# Robust Hausman test on Wooldridge `wagepan`

This validation case implements the robust Hausman test (Cameron-Trivedi 2005, Wooldridge 2010) on the wagepan dataset.

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** Balanced panel with 722 observations (36 firms × 20 years).

## Reference implementation

- **R:** `plm::phtest(fe_model, re_model, method="ht")` with cluster-robust covariance
- **Python:** Custom implementation using cluster-robust covariance matrices

## Compared quantities

- test_statistic (Hausman statistic)
- p_value
- degrees_of_freedom

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| test_statistic | 1e-4 | Robust statistic may have small numerical differences |
| p_value | 1e-4 | Conservative tolerance for p-value |
| degrees_of_freedom | 0 | Exact count |

## Notes

The robust Hausman test compares FE and RE estimators using a cluster-robust covariance matrix, making it valid under heteroskedasticity and clustering. This case uses worker-level clustering as in Wooldridge Chapter 14.

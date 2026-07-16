# Robust F-test on Wooldridge `wage1`

This validation case implements the robust F-test (Wooldridge 2010) on the wage1 dataset.

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 526 observations × 24 variables.

## Reference implementation

- **R:** `linearHypothesis()` with cluster-robust covariance (sandwich)
- **Python:** Custom Wald test with cluster-robust covariance

## Compared quantities

- test_statistic (F-statistic)
- p_value
- degrees_of_freedom_num (numerator)
- degrees_of_freedom_denom (denominator)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| test_statistic | 1e-4 | Robust statistic may have small numerical differences |
| p_value | 1e-4 | Conservative tolerance for p-value |
| degrees_of_freedom_num | 0 | Exact count |
| degrees_of_freedom_denom | 0 | Exact count |

## Notes

The robust F-test performs a joint significance test using cluster-robust covariance, making it valid under heteroskedasticity and clustering. This case tests the joint significance of experience and tenure.

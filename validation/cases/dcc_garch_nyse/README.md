# DCC-GARCH on NYSE returns

This validation case implements DCC-GARCH (Dynamic Conditional Correlation GARCH) on NYSE returns.

## Dataset

- **Name:** `wooldridge::nyse`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 551 daily observations of NYSE returns.

## Reference implementation

- **R:** `rugarch::dccfit()` or custom DCC-GARCH implementation
- **Python:** `arch` package with DCC-GARCH support

## Compared quantities

- conditional_variance_diag (diagonal elements of conditional variance matrix)
- conditional_covariance (off-diagonal conditional covariance)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| conditional_variance_diag | 1e-3 | GARCH optimization may have small numerical differences |
| conditional_covariance | 1e-3 | Correlation estimation may vary slightly |

## Notes

DCC-GARCH models both conditional variance and conditional correlation dynamics. This case uses a simplified DCC-GARCH(1,1) model on NYSE returns. Due to the complexity of DCC-GARCH estimation, the reference implementations use simplified models or approximations. The tolerances are set to account for these approximations.

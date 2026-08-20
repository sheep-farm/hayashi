# Panel Heckman selection model on simulated data

This validation case implements the Panel Heckman selection model on simulated panel data with known selection mechanism.

## Dataset

- **Name:** `simulated_panel_heckman`
- **Source:** Simulated panel DGP with selection equation
- **Licence:** MIT
- **Size:** 100 entities × 10 time periods = 1000 observations

## Reference implementation

- **R:** `sampleSelection::heckit()` with panel structure
- **Python:** Custom implementation using statsmodels

## Compared quantities

- selection_coefficients (coefficients from selection equation)
- outcome_coefficients (coefficients from outcome equation)
- inverse_mills_ratio (average IMR)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| selection_coefficients | 1e-3 | Heckman estimation may have small numerical differences |
| outcome_coefficients | 1e-3 | Two-step estimation may vary slightly |
| inverse_mills_ratio | 1e-3 | IMR computation may vary slightly |

## Notes

The Panel Heckman model corrects for sample selection bias in panel data. This case uses a simulated DGP with a selection equation based on covariates and an outcome equation that is only observed when selection = 1. The reference implementations use simplified two-step Heckman estimation due to the complexity of full panel Heckman models.

# Spatial Durbin model on simulated spatial data

This validation case implements the Spatial Durbin model on simulated spatial data with known spatial structure.

## Dataset

- **Name:** `simulated_spatial_durbin`
- **Source:** Simulated spatial DGP with SAR structure
- **Licence:** MIT
- **Size:** 100 spatial units with 5x5 grid structure

## Reference implementation

- **R:** `spdep::lagsarlm()` or `spatialreg::spatialdurbin()`
- **Python:** Custom implementation using spatial weights

## Compared quantities

- direct_effect (direct spatial effect)
- indirect_effect (indirect/spillover effect)
- total_effect (sum of direct and indirect effects)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| direct_effect | 1e-3 | Spatial estimation may have small numerical differences |
| indirect_effect | 1e-3 | Spillover effects may vary slightly |
| total_effect | 1e-3 | Total effect may vary slightly |

## Notes

The Spatial Durbin model includes both a spatial lag of the dependent variable and spatially lagged independent variables. This case uses a simple 5x5 grid with rook contiguity weights. The reference implementations use simplified spatial estimation methods due to the complexity of full spatial econometric models.

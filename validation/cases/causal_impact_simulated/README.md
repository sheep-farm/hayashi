# Causal Impact on simulated time series

This validation case implements Causal Impact (Brodersen et al. 2015) on a simulated time series with known treatment effect.

## Dataset

- **Name:** `simulated_causal_impact`
- **Source:** Simulated DGP following Brodersen et al. (2015)
- **Licence:** MIT
- **Size:** 200 time points (100 pre-treatment, 100 post-treatment)

## Reference implementation

- **R:** `CausalImpact::CausalImpact()` package
- **Python:** Custom implementation using statsmodels

## Compared quantities

- point_effect (average treatment effect during post-period)
- point_effect_lower (lower 95% CI)
- point_effect_upper (upper 95% CI)
- cumulative_effect (cumulative treatment effect)
- cumulative_effect_lower (lower 95% CI)
- cumulative_effect_upper (upper 95% CI)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| point_effect | 1e-2 | Bayesian posterior may have small MCMC differences |
| point_effect_lower | 1e-2 | CI bounds may vary |
| point_effect_upper | 1e-2 | CI bounds may vary |
| cumulative_effect | 1e-2 | Cumulative effect may vary |
| cumulative_effect_lower | 1e-2 | CI bounds may vary |
| cumulative_effect_upper | 1e-2 | CI bounds may vary |

## Notes

Causal Impact uses Bayesian structural time series to estimate the counterfactual. This case uses a simple simulated DGP with a known treatment effect of 10 units during the post-treatment period. The reference implementations use the same model specification (local level + seasonal components).

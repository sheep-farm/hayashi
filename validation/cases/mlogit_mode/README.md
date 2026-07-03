# Multinomial logit on travel mode choice data

This validation case estimates a multinomial logit model of chosen travel mode on income, wait time, vehicle cost and travel time using the `AER::TravelMode` dataset.

## Model

```
mode ~ income + wait + vcost + travel
```

## Dataset

- **Name:** `AER::TravelMode`
- **Source:** Rdatasets (`https://vincentarelbundock.github.io/Rdatasets/csv/AER/TravelMode.csv`); the reference scripts collapse the long-format choice data to one observation per individual.
- **Licence:** Public teaching dataset.
- **Size:** 210 individuals after collapsing the 840 long-format rows (4 alternatives per individual).
- **Mode encoding:** `air=1`, `train=2`, `bus=3`, `car=4`. The reference category is `car=4`.
- **Covariate handling:** `wait`, `vcost` and `travel` are alternative-specific attributes. To make them usable in a standard multinomial logit (which requires individual-specific covariates), each is averaged over the four alternatives for each individual before fitting.

## Reference implementation

- **R:** `nnet::multinom(mode ~ income + wait + vcost + travel, data = df, trace = FALSE)` with `mode` as a factor whose reference level is `car=4`
- **Python:** `statsmodels.api.MNLogit` on the collapsed data set with `mode` as a categorical whose reference level is `car=4`
- **Hayashi:** `mlogit(mode ~ income + wait + vcost + travel, df)`

## Compared quantities

- coefficients only (one set per non-reference category: 1, 2 and 3)
- keys are formatted as `{category}:{variable}` (e.g. `1:income` for `air` vs `car`)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-1 | MLE optimisation and category ordering may differ slightly across solvers |

## Notes

The `AER::TravelMode` data are provided in long format (one row per alternative). The reference scripts collapse them to one row per individual and average the alternative-specific attributes (wait, vcost, travel) over the four alternatives so that they become individual-specific covariates suitable for a standard multinomial logit.

# Propensity score matching on Wooldridge jtrain3

This validation case estimates the average treatment effect on the treated (ATT)
of job training (`train`) on real earnings in 1978 (`re78`) using 1:1
nearest-neighbor propensity score matching with a caliper.

## Dataset

- **Name:** `wooldridge::jtrain3`
- **Source:** Python `wooldridge` package; also available in R via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Variables:**
  - Outcome: `re78`
  - Treatment: `train`
  - Covariates: `age`, `educ`, `black`, `hisp`, `married`, `unem74`,
    `unem75`, `re74`, `re75` (note: `nodegree` is unavailable in `jtrain3`).

## Model

```
psm(re78 ~ train + age + educ + black + hisp + married +
            unem74 + unem75 + re74 + re75,
    df, k=1, caliper=0.2, boot=200)
```

## Reference implementation

- **Python:** reference implementation using `statsmodels.logit` for the propensity score, `sklearn.neighbors.NearestNeighbors` for 1:1 nearest-neighbor matching, and a bootstrap SE with 200 replications.
- **R:** reference implementation using base R (`glm(family = binomial)` for the propensity score; manual 1:1 nearest-neighbor matching; bootstrap SE with 200 replications). Both references are run by the validation runner.

## Compared quantities

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients (ATT) | 1e-1 | Different matching algorithms may differ slightly. |
| standard_errors (ATT) | 5e-1 | Bootstrap SE is simulation-sensitive. |

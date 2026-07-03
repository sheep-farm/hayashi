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

- **Python:** primary reference implementation (`statsmodels.logit` for propensity score; `sklearn.NearestNeighbors` for 1:1 nearest-neighbor matching; bootstrap SE with 200 replications).
- **R:** a reference script is provided and implements the matching manually in base R (`glm(family = binomial)` for propensity score; manual 1:1 nearest-neighbor matching; bootstrap SE with 200 replications). It does not require the `MatchIt` package. When R is unavailable, the validation relies on the Python reference.

## Compared quantities

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients (ATT) | 1e-1 | Different matching algorithms may differ slightly. |
| standard_errors (ATT) | 5e-1 | Bootstrap SE is simulation-sensitive. |

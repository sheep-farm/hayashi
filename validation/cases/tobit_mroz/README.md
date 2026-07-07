# Tobit hours-worked model on Wooldridge `mroz`

This validation case estimates a Tobit regression of hours worked, left-censored at zero.

## Model

```
hours ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6
```

with left censoring at `hours = 0`.

## Dataset

- **Name:** `wooldridge::mroz`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 753 observations × 22 variables.

## Reference implementation

- **R (active reference):** `AER::tobit(hours ~ ..., data = mroz, left = 0)`
- **Python (diagnostic only):** Custom maximum-likelihood Tobit implementation via `scipy.optimize` (log-likelihood with normal censoring). It is retained for investigation, but is not the active validation reference because it can converge to a nearby solution with a modest intercept delta.
- **Hayashi:** `tobit(hours ~ ..., df, ll=0)`

## Compared quantities

- coefficients
- standard errors (AER's default covariance estimate)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | Hayashi and `AER::tobit` match at displayed precision |
| standard_errors | 1e-3 | Hayashi and `AER::tobit` match at displayed precision |

## Notes

Issue [#43](https://github.com/sheep-farm/hayashi/issues/43) records the investigation that led to this case using R/AER as the active reference. The earlier loose tolerances were needed only while the case was effectively comparing against the custom Python MLE.

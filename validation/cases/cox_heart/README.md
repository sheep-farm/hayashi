# Cox proportional hazards on heart transplant survival

This validation case estimates a Cox proportional hazards model for survival time after heart transplant.

## Model

```
h(time) ~ age
```

where `time` is the survival time and `censored` indicates right censoring.

## Dataset

- **Name:** `statsmodels::heart`
- **Source:** `statsmodels.datasets` (Python) and Rdatasets mirror (R).
- **Licence:** Public teaching dataset.
- **Size:** 69 observations × 3 variables.

## Reference implementation

- **R:** `survival::coxph(Surv(death, censored) ~ age, data = heart)`
- **Python:** `statsmodels.duration.hazard_regression.PHReg("death ~ age", data=heart, status=heart["censored"]).fit()`
- **Hayashi:** `cox(age, df, time=death)`

## Compared quantities

- coefficients (log hazard ratios)
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | Cox partial likelihood should match closely |
| standard_errors | 1e-3 | Same tolerance as coefficients |

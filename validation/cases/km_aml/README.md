# Kaplan-Meier survival checkpoints on acute myelogenous leukaemia remission data

This validation case estimates a marginal Kaplan-Meier survival curve for
time to relapse in acute myelogenous leukaemia remission data.

## Dataset

- **Name:** `survival::aml`
- **Source:** R `survival` package; originally cited to Miller (1997).
- **Licence:** LGPL (>= 2), inherited from the R `survival` package.
- **Size:** 23 observations x 2 validation variables (`time`, `event`).

The data generator writes a deterministic two-column CSV projection and checks
its SHA-256 digest before references or Hayashi consume it.

## Analysis

The validation compares the pooled Kaplan-Meier survival curve, without group
stratification. It checks right-continuous survival probabilities at:

```
t10, t20, t30, t40, t50, t60, t70
```

The case deliberately does not validate confidence intervals. Hayashi currently
prints survival probabilities to four decimal places, so the validation
tolerance is set to `1e-4`.

## Reference implementation

- **R:** `survival::survfit(Surv(time, event) ~ 1, data = aml)`
- **Python:** `statsmodels.duration.survfunc.SurvfuncRight(time, event)`
- **Hayashi:** `km(time, event, df)`

Both reference scripts assert fixed checkpoint values at absolute tolerance
`1e-12` before emitting JSON.

## Compared quantities

- survival probabilities at `t10`, `t20`, `t30`, `t40`, `t50`, `t60`, and `t70`

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| survival_probabilities | 1e-4 | Hayashi's printable Kaplan-Meier table displays four decimal places. |

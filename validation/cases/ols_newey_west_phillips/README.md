# ols_newey_west_phillips

Validates OLS coefficients and Newey-West/HAC standard errors on the
Wooldridge `phillips` time-series dataset.

## Dataset

- **Name:** `wooldridge::phillips`
- **Source:** R package `wooldridge`; Python package `wooldridge` with an
  Rdatasets CSV fallback
- **Licence:** public teaching dataset
- **Size:** 56 observations before dropping rows with missing model variables

## Analysis

The case estimates the expectations-augmented Phillips curve from Wooldridge
Chapter 11, Example 11.5:

```text
cinf ~ unem
```

This isolates the covariance path for Hayashi's public `nw=` option:

```hayashi
ols(cinf ~ unem, df, nw=4)
```

OLS coefficients should match the homoskedastic fit. The validation target is
the Newey-West/HAC standard error calculation.

## Reference implementation

- **R:** `lm`, with Newey-West covariance computed manually in base R.
- **Python:** `statsmodels.formula.api.ols`, with Newey-West covariance
  computed manually in NumPy from the fitted design matrix and residuals.

The references implement Hayashi/Greeners' declared convention directly:

```text
V_NW = c * (X'X)^-1 * [Omega_0 + sum_l w_l(Omega_l + Omega_l')] * (X'X)^-1

Omega_0 = X' diag(e_i^2) X
Omega_l = sum_{t=l+1}^n e_t e_{t-l} x_t x_{t-l}'
w_l = 1 - l / (L + 1)
c = n / (n - k)
```

where `L = 4`, `n` is the model-frame observation count, and `k` is the number
of estimated coefficients including the intercept.

## Compared quantities

- coefficients
- standard errors

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---:|---|
| coefficients | `1e-6` | OLS coefficients are deterministic closed-form quantities. |
| standard_errors | `1e-6` | The HAC covariance is deterministic once the kernel, lag length, and correction are fixed. |

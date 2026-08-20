# ols_hc3_wooldridge_wage1

Validates OLS coefficients and HC3 heteroskedasticity-robust standard errors
on the Wooldridge `wage1` dataset.

## Dataset

- **Name:** `wooldridge::wage1`
- **Source:** R package `wooldridge`; Python package `wooldridge` with an
  Rdatasets CSV fallback
- **Licence:** public teaching dataset
- **Size:** 526 observations, 24 variables

## Analysis

The case estimates a log-wage equation:

```text
lwage ~ educ + exper + tenure
```

This isolates the covariance path for Hayashi's public `cov=HC3` option:

```hayashi
ols(lwage ~ educ + exper + tenure, df, cov=HC3)
```

OLS coefficients should match the homoskedastic fit. The validation target is
the HC3 standard error calculation.

## Reference implementation

- **R:** `lm`, with HC3 covariance computed manually from the hat values.
- **Python:** `statsmodels.formula.api.ols(...).fit(cov_type="HC3")`.

The R reference uses the standard HC3 sandwich:

```text
V_HC3 = (X'X)^-1 X' diag(e_i^2 / (1 - h_i)^2) X (X'X)^-1
```

where `e_i` are OLS residuals and `h_i` are diagonal leverage values from the
hat matrix.

## Compared quantities

- coefficients
- standard errors

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---:|---|
| coefficients | `1e-6` | OLS coefficients are deterministic closed-form quantities. |
| standard_errors | `1e-6` | HC3 covariance is deterministic for a fixed model frame. |

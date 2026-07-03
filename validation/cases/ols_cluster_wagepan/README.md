# ols_cluster_wagepan

Validates OLS coefficients and one-way cluster-robust standard errors on the
Wooldridge `wagepan` panel dataset.

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`; Python package `wooldridge` with an
  Rdatasets CSV fallback
- **Licence:** public teaching dataset
- **Size:** 4,360 observations, 44 variables

## Analysis

The case estimates a pooled log-wage equation and clusters standard errors by
worker id:

```text
lwage ~ educ + exper + expersq + union + married
clustered by nr
```

This intentionally validates inference rather than only point estimates. It
covers Hayashi's public `cluster=` covariance option for OLS on a real
panel-style dataset with 545 worker clusters.

## Reference implementation

- **R:** `lm`, with one-way cluster-robust covariance computed manually in
  base R.
- **Python:** `statsmodels.formula.api.ols(...).fit(cov_type="cluster")` with
  small-sample correction enabled.

The R implementation applies the same finite-sample correction convention used
by statsmodels and Hayashi:

```text
(G / (G - 1)) * ((N - 1) / (N - K))
```

where `G` is the number of clusters, `N` is the number of observations, and
`K` is the number of estimated coefficients including the intercept.

## Compared quantities

- coefficients
- standard errors

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---:|---|
| coefficients | `1e-6` | OLS coefficients are deterministic closed-form quantities. |
| standard_errors | `1e-6` | Clustered covariance is deterministic once the finite-sample correction is fixed. |

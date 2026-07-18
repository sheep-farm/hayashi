# ols_twoway_cluster_wagepan

Validates OLS coefficients and two-way clustered standard errors on the
Wooldridge `wagepan` panel dataset.

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`; Python package `wooldridge` with an
  Rdatasets CSV fallback
- **Licence:** public teaching dataset
- **Size:** 4,360 observations, 44 variables

## Analysis

The case estimates the same pooled log-wage equation used by the one-way
clustered OLS validation case, then clusters standard errors by worker and year:

```text
lwage ~ educ + exper + expersq + union + married
clustered by nr and year
```

This isolates the covariance path for Hayashi's public `cluster2=` option:

```hayashi
ols(lwage ~ educ + exper + expersq + union + married, df,
    cluster=nr, cluster2=year)
```

The `year` dimension has only eight clusters. This case is therefore an
implementation validation of the advertised option, not an empirical
recommendation that this specification is adequate for applied inference.

## Reference implementation

- **R:** `lm`, with two-way clustered covariance computed manually in base R.
- **Python:** `statsmodels.formula.api.ols`, with two-way clustered covariance
  computed manually in NumPy/Pandas.

Packaged two-way cluster defaults differ in finite-sample corrections across
software. The references therefore implement Hayashi's declared convention
directly:

```text
V_2way = c * (X'X)^-1 * (M_nr + M_year - M_nr_year) * (X'X)^-1

M_g = sum_g (X_g' u_g)(X_g' u_g)'

c = (min(G_nr, G_year) / (min(G_nr, G_year) - 1)) * ((N - 1) / (N - K))
```

where `M_nr_year` is computed on the intersection clusters `(nr, year)`, `N` is
the model-frame observation count, and `K` is the number of estimated
coefficients including the intercept.

## Compared quantities

- coefficients
- standard errors

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---:|---|
| coefficients | `1e-6` | OLS coefficients are deterministic closed-form quantities. |
| standard_errors | `1e-6` | The two-way covariance is deterministic once the correction convention is fixed. |

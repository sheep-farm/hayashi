# iv_robust_mroz

Validates IV/2SLS coefficients and heteroskedasticity-robust standard errors on
the Wooldridge `mroz` dataset.

## Dataset

- **Name:** `wooldridge::mroz`
- **Source:** R package `wooldridge`; Python package `wooldridge` with an
  Rdatasets CSV fallback
- **Licence:** public teaching dataset

## Analysis

The case estimates the Wooldridge Chapter 15 returns-to-schooling IV equation:

```text
lwage ~ educ + exper + expersq
instruments: fatheduc, motheduc, exper, expersq
```

with Hayashi's public robust covariance option:

```hayashi
iv(lwage ~ educ + exper + expersq,
   ~ fatheduc + motheduc + exper + expersq,
   df,
   cov=robust)
```

This isolates the IV HC1 covariance path. The point estimates should match the
existing non-robust IV case; the validation target is the robust standard-error
calculation.

## Reference Implementation

Both references compute the Hayashi/Greeners convention explicitly:

```text
X_hat = Z (Z'Z)^-1 Z'X
beta = (X_hat'X_hat)^-1 X_hat'y
u = y - X beta
V_HC1 = [n / (n-k)] (X_hat'X_hat)^-1 X_hat' diag(u^2) X_hat (X_hat'X_hat)^-1
```

- **R:** manual matrix calculation in base R.
- **Python:** manual matrix calculation in NumPy/Pandas.

## Compared Quantities

- coefficients
- robust standard errors

## Tolerances and Rationale

| Quantity | Tolerance | Rationale |
|---|---:|---|
| coefficients | `1e-4` | Hayashi IV text export displays coefficients to four decimals. |
| standard_errors | `1e-4` | Hayashi IV text export displays standard errors to four decimals. |

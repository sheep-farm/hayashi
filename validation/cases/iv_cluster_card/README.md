# iv_cluster_card

Validates IV/2SLS coefficients and one-way clustered standard errors on the
Wooldridge `card` dataset.

## Dataset

- **Name:** `wooldridge::card`
- **Source:** R package `wooldridge`; Python package `wooldridge` with an
  Rdatasets CSV fallback
- **Licence:** public teaching dataset
- **Size:** 3,010 observations

## Analysis

The case estimates the Card returns-to-schooling IV equation:

```text
lwage ~ educ + exper + expersq + black + south + smsa
instruments: nearc4, exper, expersq, black, south, smsa
```

and clusters standard errors by Census region, derived from the mutually
exclusive `reg661` through `reg669` indicators:

```hayashi
iv(lwage ~ feduc + fexper + fexpersq + fblack + fsouth + fsmsa,
   ~ fnearc4 + fexper + fexpersq + fblack + fsouth + fsmsa,
   df,
   cluster=region)
```

The `f*` variables are numeric copies used to avoid CSV integer/boolean
inference. This isolates the IV clustered covariance path. The point estimates
should match the existing non-robust Card IV case; the validation target is the
clustered standard-error calculation.

## Reference Implementation

Both references compute the Hayashi/Greeners convention explicitly:

```text
X_hat = Z (Z'Z)^-1 Z'X
beta = (X_hat'X_hat)^-1 X_hat'y
u = y - X beta
V_cluster = [G/(G-1)] [(n-1)/(n-k)] (X_hat'X_hat)^-1
            [sum_g X_hat_g' u_g u_g' X_hat_g]
            (X_hat'X_hat)^-1
```

- **R:** manual matrix calculation in base R.
- **Python:** manual matrix calculation in NumPy/Pandas.

## Compared Quantities

- coefficients
- one-way clustered standard errors

## Tolerances and Rationale

| Quantity | Tolerance | Rationale |
|---|---:|---|
| coefficients | `1e-4` | Hayashi IV text export displays coefficients to four decimals. |
| standard_errors | `1e-4` | Hayashi IV text export displays standard errors to four decimals. |

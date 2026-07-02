# IV returns to schooling on Wooldridge `card`

This validation case estimates the classic Card (1995) returns-to-schooling
model using instrumental variables.

## Model

```
lwage ~ educ + exper + expersq + black + south + smsa
```

where `educ` is endogenous and instrumented by `nearc4` (grew up near a
four-year college).

## Dataset

- **Name:** `wooldridge::card`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 3,010 observations × 34 variables (after dropping missing values).

## Reference implementation

- **R:** `AER::ivreg(lwage ~ educ + ... | nearc4 + ..., data = card)`
- **Python:** `linearmodels.IV2SLS(...)` with `nearc4` as instrument for `educ`.
- **Hayashi:** `iv(lwage ~ educ + ..., ~ nearc4 + ..., df)`.

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | IV/2SLS has more numerical variation across implementations |
| standard_errors | 1e-4 | Same tolerance as coefficients |

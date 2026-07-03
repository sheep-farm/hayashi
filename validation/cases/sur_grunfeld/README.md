# Seemingly Unrelated Regressions on Wooldridge `grunfeld`

This validation case estimates a two-equation SUR (Zellner FGLS) on the Grunfeld investment data.

## Model

Two equations estimated jointly:

```
value ~ inv + capital
inv ~ value + capital
```

The Wooldridge `grunfeld` data names the investment variable `inv`.

## Dataset

- **Name:** `wooldridge::grunfeld`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 200 observations (10 firms × 20 years).

## Reference implementation

- **R:** `systemfit::systemfit(list(eq1 = value ~ inv + capital, eq2 = inv ~ value + capital), data = df, method = "SUR")` (requires the `systemfit` package)
- **Python:** manual Zellner FGLS using OLS residuals and block-diagonal GLS
- **Hayashi:** `sur(df, value ~ inv + capital, inv ~ value + capital)`

## Compared quantities

- coefficients only
- keys are formatted as `{equation}:{variable}` (e.g. `value:inv`)

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-1 | FGLS estimates and covariance approximations may differ slightly across implementations |

## Notes

The R reference requires the `systemfit` package. If it is not installed the R script is skipped and the Python reference is used for comparison.

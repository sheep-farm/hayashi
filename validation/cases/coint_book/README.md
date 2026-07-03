# Bivariate cointegrated system from the Hayashi book

This validation case estimates a VECM(1) on a simulated bivariate cointegrated system.

## Status

`active`

## Model

```
x_t = x_{t-1} + e1_t
y_t = 2 * x_t + e2_t
```

Equivalently, `y_t - 2*x_t` is a stationary cointegration error. With the Johansen normalization used by Hayashi, the long-run cointegration vector should be close to `[1, -2]`.

## Dataset

- **Name:** `simulated_cointegrated`
- **Source:** DGP from `book_pt_BR/codes/29_coint.hay`
- **Size:** 300 observations
- **Variables:** `y`, `x`

## Reference implementation

- **R:** manual Johansen ML procedure implemented with base R.
- **Python:** manual Johansen ML procedure implemented with NumPy.
- **Hayashi:** `vecm(df, y, x, lags=1)`.

## Compared quantities

- coefficients
- standard_errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-2 | Johansen ML estimates should match closely for the beta and alpha coefficients. |
| standard_errors | 5e-1 | Bootstrap SEs from Hayashi (`with_inference(200)`) are compared against simple reference SEs. The tolerance is loose because the reference uses approximations rather than the same bootstrap. |

## Reference standard errors

- **Alpha:** OLS conditional SEs from the regression of each `Δy_jt` on the estimated cointegration term `β' y_{t-1}` (orthogonal to the constant).
- **Beta:** Rough Engle-Granger/OLS proxies from the static long-run regression `y ~ x` (with intercept). The intercept SE is used as a proxy for `beta_1_y1` and the slope SE as a proxy for `beta_1_y2`. These are not the exact Johansen asymptotic SEs, but they are easy to compute and stay within the declared tolerance.

## Output format

Hayashi exports a plain-text table (`txt`) containing the cointegration vector `beta_1_y1`, `beta_1_y2` and the adjustment coefficients `alpha_1_y1`, `alpha_1_y2`, together with their standard errors.

# Bivariate cointegrated system from the Hayashi book

This validation case estimates a VECM(1) on a simulated bivariate cointegrated system.

## Status

`active`

## Model

```
x_t = x_{t-1} + e1_t
y_t = 2 * x_t + e2_t
```

Equivalently, `y_t - 2*x_t` is a stationary cointegration error. The long-run
cointegration vector is proportional to `[1, -2]`; the references use
Cholesky-based Johansen scaling, not a first coefficient fixed at one.

## Dataset

- **Name:** `simulated_cointegrated`
- **Source:** DGP from `book_pt_BR/codes/29_coint.hay`
- **Size:** 300 observations
- **Variables:** `y`, `x`

## Reference implementation

- **R:** manual Johansen ML procedure implemented with base R.
- **Python:** manual Johansen ML procedure implemented with NumPy.
- **Hayashi:** `vecm(df, y, x, lags=1)`.

## Reference evidence

For both R and Python, `coefficients` is `convention-matched`: the manual
Johansen procedures align rank 1, lag order 1, intercept residualisation and
Cholesky-based beta normalisation with corresponding alpha scaling.
This classification does not certify independent provenance.

`standard_errors` is `behavioural-proxy`. These legacy comparisons are retained
diagnostics, not validation of Hayashi's bootstrap uncertainty. Neither their
recorded pass nor their existing loose tolerance justifies new loose-SE
comparisons as estimator inference validation. See
[evidence classes](../../README.md#reference-evidence-classes).

## Compared quantities

- coefficients
- standard_errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-2 | Johansen ML estimates should match closely for the beta and alpha coefficients. |
| standard_errors | 5e-1 | Retained legacy proxy diagnostic comparing Hayashi bootstrap SEs (`with_inference(200)`) with different reference SE constructions; not validation of bootstrap uncertainty. |

## Reference standard errors

- **Alpha:** OLS conditional SEs from the regression of each `Δy_jt` on the estimated cointegration term `β' y_{t-1}` (orthogonal to the constant).
- **Beta:** Rough Engle-Granger/OLS proxies from the static long-run regression `y ~ x` (with intercept). The intercept SE is used as a proxy for `beta_1_y1` and the slope SE as a proxy for `beta_1_y2`. These are neither Johansen asymptotic SEs nor the bootstrap uncertainty produced by Hayashi. Agreement within the tolerance cannot establish inference equivalence.

## Output format

Hayashi exports a plain-text table (`txt`) containing the cointegration vector `beta_1_y1`, `beta_1_y2` and the adjustment coefficients `alpha_1_y1`, `alpha_1_y2`, together with their standard errors.

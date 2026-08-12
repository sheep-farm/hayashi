# lowess_simulated

LOWESS smoothing validation on simulated sine data.

- DGP: `y = sin(x) + N(0, 0.2)` with 200 observations.
- Hayashi: `lowess(df, y, x, frac=0.4, it=3)` then `predict df yhat = m, "smoothed"`.
- R reference: `lowess(df$x, df$y, f=0.4, iter=3)` interpolated back to original `x`.
- Python reference: `statsmodels.nonparametric.smoothers_lowess.lowess(..., frac=0.4, it=3, return_sorted=False)`.
- Output: `mean_yhat`, `yhat_first`, `yhat_mid`, `yhat_last`.

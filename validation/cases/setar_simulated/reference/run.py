import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].values
n = len(y)
delay = 1

# Construct lagged data: y_t = c + phi * y_{t-delay} + e
y_lag = y[:-delay]
y_t = y[delay:]

# Grid search over candidate thresholds (quantiles of the threshold variable)
thresh_var = y_lag
sorted_th = np.sort(np.unique(thresh_var))
lo = sorted_th[int(0.15 * len(sorted_th))]
hi = sorted_th[int(0.85 * len(sorted_th))]
candidates = np.linspace(lo, hi, 50)

best = None
best_rss = np.inf
for th in candidates:
    low_idx = thresh_var < th
    high_idx = ~low_idx
    rss = 0.0
    res = {}
    for name, idx in [("low", low_idx), ("high", high_idx)]:
        x = np.column_stack([np.ones(idx.sum()), y_lag[idx]])
        yy = y_t[idx]
        beta = np.linalg.lstsq(x, yy, rcond=None)[0]
        pred = x @ beta
        rss += np.sum((yy - pred) ** 2)
        res[name] = (beta, x)
    if rss < best_rss:
        best_rss = rss
        best = (th, res)

threshold, res = best
beta_low, X_low = res["low"]
beta_high, X_high = res["high"]

# Standard errors from OLS residual variance
resid_low = X_low @ beta_low - y_t[thresh_var < threshold]
resid_high = X_high @ beta_high - y_t[thresh_var >= threshold]
sigma2_low = np.sum(resid_low ** 2) / (len(resid_low) - X_low.shape[1])
sigma2_high = np.sum(resid_high ** 2) / (len(resid_high) - X_high.shape[1])
se_low = np.sqrt(np.diag(np.linalg.inv(X_low.T @ X_low)) * sigma2_low)
se_high = np.sqrt(np.diag(np.linalg.inv(X_high.T @ X_high)) * sigma2_high)

result = {
    "coefficients": {
        "low_const": float(beta_low[0]),
        "low_AR.L1": float(beta_low[1]),
        "high_const": float(beta_high[0]),
        "high_AR.L1": float(beta_high[1]),
    },
    "standard_errors": {
        "low_const": float(se_low[0]),
        "low_AR.L1": float(se_low[1]),
        "high_const": float(se_high[0]),
        "high_AR.L1": float(se_high[1]),
    },
}

print(json.dumps(result, indent=2))

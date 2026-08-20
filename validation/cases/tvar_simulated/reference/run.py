import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y1 = df["y1"].values
y2 = df["y2"].values
q = df["q"].values

y1_t = y1[1:]
y2_t = y2[1:]
y1_l1 = y1[:-1]
y2_l1 = y2[:-1]
q_l1 = q[:-1]

sorted_q = np.sort(np.unique(q_l1))
lo = sorted_q[int(0.15 * len(sorted_q))]
hi = sorted_q[int(0.85 * len(sorted_q))]
candidates = np.linspace(lo, hi, 50)

best_c = None
best_rss = np.inf
for c in candidates:
    low = q_l1 < c
    high = ~low
    rss = 0.0
    for y_t in (y1_t, y2_t):
        for idx in (low, high):
            X = np.column_stack([y1_l1[idx], y2_l1[idx]])
            beta = np.linalg.lstsq(X, y_t[idx], rcond=None)[0]
            resid = y_t[idx] - X @ beta
            rss += np.sum(resid ** 2)
    if rss < best_rss:
        best_rss = rss
        best_c = c

c = best_c
low = q_l1 < c
high = ~low


def ols_noint(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sigma2 = np.sum(resid ** 2) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sigma2)
    return beta, se


coefs = {}
ses = {}
for regime, idx, y_t, eq in [
    ("low", low, y1_t, "y1"),
    ("low", low, y2_t, "y2"),
    ("high", high, y1_t, "y1"),
    ("high", high, y2_t, "y2"),
]:
    X = np.column_stack([y1_l1[idx], y2_l1[idx]])
    beta, se = ols_noint(y_t[idx], X)
    for i, var in enumerate(["L1.y1", "L1.y2"]):
        key = f"{regime}_{eq}_{var}"
        coefs[key] = float(beta[i])
        ses[key] = float(se[i])

result = {"coefficients": coefs, "standard_errors": ses}
print(json.dumps(result, indent=2))

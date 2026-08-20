import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
ids = df["id"].values
y1 = df["y1"].values
y2 = df["y2"].values

panels = np.unique(ids)
T = int(np.bincount(ids).max())
Y1 = np.zeros((len(panels), T))
Y2 = np.zeros((len(panels), T))
for i, p in enumerate(panels):
    idx = ids == p
    Y1[i, :] = y1[idx][:T]
    Y2[i, :] = y2[idx][:T]

# Within transformation: remove time mean per panel
Y1w = Y1 - Y1.mean(axis=1, keepdims=True)
Y2w = Y2 - Y2.mean(axis=1, keepdims=True)

# Regress y1 on L.y1 and L.y2, y2 on L.y1 and L.y2 (within estimator)
# Use t=1..T-1, with t-1 lags
Y1_l1 = Y1w[:, :-1]
Y2_l1 = Y2w[:, :-1]
y1_dep = Y1w[:, 1:]
y2_dep = Y2w[:, 1:]

X = np.column_stack([Y1_l1.ravel(), Y2_l1.ravel()])

def ols_no_intercept(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    nobs = len(y)
    k = X.shape[1]
    sigma2 = np.sum(resid ** 2) / (nobs - k)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sigma2)
    return beta, se

b_y1, se_y1 = ols_no_intercept(y1_dep.ravel(), X)
b_y2, se_y2 = ols_no_intercept(y2_dep.ravel(), X)

result = {
    "coefficients": {
        "y1_L1.y1": float(b_y1[0]),
        "y1_L1.y2": float(b_y1[1]),
        "y2_L1.y1": float(b_y2[0]),
        "y2_L1.y2": float(b_y2[1]),
    },
    "standard_errors": {
        "y1_L1.y1": float(se_y1[0]),
        "y1_L1.y2": float(se_y1[1]),
        "y2_L1.y1": float(se_y2[0]),
        "y2_L1.y2": float(se_y2[1]),
    },
}

print(json.dumps(result, indent=2))

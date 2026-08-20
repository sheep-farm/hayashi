import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].values
x = df["x"].values
q = df["q"].values
ids = df["id"].values
n = len(y)

# Within demeaning
y = y.copy()
x = x.copy()
q = q.copy()
for vec in [y, x, q]:
    for uid in np.unique(ids):
        mask = ids == uid
        vec[mask] = vec[mask] - vec[mask].mean()

# Grid over c and gamma
cs = np.linspace(q.min(), q.max(), 41)
gammas = np.linspace(0.1, 15.0, 61)

best = (1.0, 0.0, np.inf)
for gamma in gammas:
    for c in cs:
        g = 1.0 / (1.0 + np.exp(-gamma * (q - c)))
        Z = np.column_stack([x, x * g])
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        res = y - Z @ beta
        sse = np.sum(res ** 2)
        if sse < best[2]:
            best = (gamma, c, sse)

gamma, c, _ = best
g = 1.0 / (1.0 + np.exp(-gamma * (q - c)))
Z = np.column_stack([x, x * g])
beta = np.linalg.lstsq(Z, y, rcond=None)[0]
res = y - Z @ beta
sigma2 = np.sum(res ** 2) / (n - 2 - 2)
cov = np.linalg.inv(Z.T @ Z + 1e-8 * np.eye(2)) * sigma2
se = np.sqrt(np.diag(cov))

result = {
    "coefficients": {
        "gamma": float(gamma),
        "c": float(c),
        "beta0_x": float(beta[0]),
        "beta1_x": float(beta[1]),
    },
    "standard_errors": {
        "gamma": 0.0,
        "c": 0.0,
        "beta0_x": float(se[0]),
        "beta1_x": float(se[1]),
    },
}

print(json.dumps(result, indent=2))

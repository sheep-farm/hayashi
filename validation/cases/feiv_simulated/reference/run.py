import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

# Within transformation by entity
vars = ["y", "x", "z"]
dm = df.groupby("id")[vars].transform(lambda s: s - s.mean())

Y = dm["y"].to_numpy()
X = dm[["x"]].to_numpy()
Z = dm[["z"]].to_numpy()
n = len(Y)
G = df["id"].nunique()

# First stage: within-X on within-Z
Xhat = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)

# Second stage: within-Y on fitted within-X
beta = np.linalg.solve(Xhat.T @ X, Xhat.T @ Y)

# Residuals using the observed (within) endogenous regressor
resid = Y - X @ beta
k = X.shape[1]
df_resid = n - k - (G - 1)
if df_resid <= 0:
    raise ValueError("FE-IV reference has no residual degrees of freedom")

sigma2 = float(resid @ resid) / df_resid
cov = sigma2 * np.linalg.solve(Xhat.T @ X, np.eye(k))
se = np.sqrt(np.diag(cov))

result = {
    "coefficients": {"x": float(beta[0])},
    "standard_errors": {"x": float(se[0])},
}

print(json.dumps(result, indent=2))

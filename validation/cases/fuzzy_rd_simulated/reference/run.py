import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
h = 0.5
sub = df[np.abs(df["x"]) <= h].copy()
x = sub["x"].values
T = (sub["x"] >= 0).astype(float).values
w = (1 - np.abs(x) / h).clip(min=0)
sqrt_w = np.sqrt(w)
Y = sub["y"].values
D = sub["d"].values
xT = x * T

X_exo = np.column_stack([np.ones(len(x)), x, xT])
Z = np.column_stack([np.ones(len(x)), x, xT, T])
X = np.column_stack([np.ones(len(x)), x, xT, D])

Y_w = Y * sqrt_w
X_w = X * sqrt_w[:, None]
Z_w = Z * sqrt_w[:, None]

Pz = Z_w @ np.linalg.pinv(Z_w.T @ Z_w) @ Z_w.T
beta = np.linalg.pinv(X_w.T @ Pz @ X_w) @ (X_w.T @ Pz @ Y_w)
tau = float(beta[-1])

result = {
  "coefficients": {"tau": tau},
  "standard_errors": {"tau": float("nan")}
}
print(json.dumps(result))

import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
W = np.array(pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "W.csv", header=None))

y = df["y"].values
x = df["x"].values
n = len(y)

# Design matrix with intercept and x (plus spatial lag of x)
WX = W.dot(x)
X = np.column_stack([np.ones(n), x, WX, W.dot(x)])

# SDM: y = rho W y + X beta + W X theta + e
# => (I - rho W) y = [1, x, Wx, W^2 x] gamma + e (but W^2 x is redundant; use x and Wx)
# Concentrated SSE for rho
w_eigvals = np.linalg.eigvals(W).real

def sdm_sse(rho):
    A = np.eye(n) - rho * W
    y_star = A.dot(y)
    Z = np.column_stack([np.ones(n), x, WX])
    gamma = np.linalg.lstsq(Z, y_star, rcond=None)[0]
    res = y_star - Z.dot(gamma)
    return np.sum(res ** 2)

lambdas = np.linspace(-0.99, 0.99, 199)
sse = np.array([sdm_sse(l) for l in lambdas])
best_rho = lambdas[np.argmin(sse)]

result = {
    "rho": float(best_rho),
}

print(json.dumps(result, indent=2))

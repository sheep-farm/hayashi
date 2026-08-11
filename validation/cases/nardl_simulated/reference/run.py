import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].values
x = df["x"].values
t = len(y)
lags = 1

# Decompose x into positive/negative partial sums
dx = np.diff(x)
x_pos = np.zeros(t)
x_neg = np.zeros(t)
cum_pos = 0.0
cum_neg = 0.0
x_pos[0] = x[0]
x_neg[0] = x[0]
for i in range(1, t):
    if dx[i-1] > 0:
        cum_pos += dx[i-1]
    else:
        cum_neg += dx[i-1]
    x_pos[i] = x[0] + cum_pos
    x_neg[i] = x[0] + cum_neg

# Build ECM regression
n_eff = t - lags - 1
n_reg = 4 + lags * 3
Z = np.zeros((n_eff, n_reg))
dy = np.zeros(n_eff)

for i in range(n_eff):
    t_i = lags + 1 + i
    dy[i] = y[t_i] - y[t_i - 1]
    Z[i, 0] = 1.0
    Z[i, 1] = y[t_i - 1]
    Z[i, 2] = x_pos[t_i - 1]
    Z[i, 3] = x_neg[t_i - 1]
    for j in range(lags):
        Z[i, 4 + j] = y[t_i - j - 1] - y[t_i - j - 2]
        Z[i, 4 + lags + j] = x_pos[t_i - j - 1] - x_pos[t_i - j - 2]
        Z[i, 4 + 2 * lags + j] = x_neg[t_i - j - 1] - x_neg[t_i - j - 2]

beta = np.linalg.lstsq(Z, dy, rcond=None)[0]
resid = dy - Z @ beta
sigma2 = np.sum(resid ** 2) / (n_eff - n_reg)
cov = np.linalg.inv(Z.T @ Z + 1e-8 * np.eye(n_reg)) * sigma2
se = np.sqrt(np.diag(cov))

names = ["const", "y_{t-1}", "x^+_{t-1}", "x^-_{t-1}", "Dy_{t-1}", "Dx^+_{t-1}", "Dx^-_{t-1}"]

result = {
    "coefficients": {n: float(beta[i]) for i, n in enumerate(names)},
    "standard_errors": {n: float(se[i]) for i, n in enumerate(names)},
}

print(json.dumps(result, indent=2))

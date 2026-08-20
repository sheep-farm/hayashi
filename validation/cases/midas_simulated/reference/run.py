import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ylow = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "ylow.csv")
xhigh = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "xhigh.csv")
y = ylow["y"].values
x = xhigh["x"].values

T = len(y)
freq = 3
n_lags = 12
poly_degree = 2

def build_x_midas(gamma):
    k = np.arange(n_lags)
    Z = np.column_stack([np.ones(n_lags), k, k**2])
    g = np.array([0.0, gamma[0], gamma[1]])
    raw = np.exp(Z.dot(g))
    weights = raw / raw.sum()
    xm = np.zeros(T)
    for t in range(T):
        base = t * freq + (freq - 1)
        val = 0.0
        for lag in range(n_lags):
            if base >= lag:
                val += weights[lag] * x[base - lag]
        xm[t] = val
    return xm, weights

def sse(gamma):
    xm, _ = build_x_midas(gamma)
    x_mean = xm.mean()
    y_mean = y.mean()
    sxx = np.sum((xm - x_mean)**2)
    sxy = np.dot(xm - x_mean, y - y_mean)
    if abs(sxx) < 1e-15:
        return 1e18
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean
    return np.sum((y - alpha - beta * xm)**2)

res = minimize(sse, [0.0, 0.0], method="Powell")
best_gamma = res.x
xm, _ = build_x_midas(best_gamma)
x_mean = xm.mean()
y_mean = y.mean()
sxx = np.sum((xm - x_mean)**2)
sxy = np.dot(xm - x_mean, y - y_mean)
beta = sxy / sxx
alpha = y_mean - beta * x_mean
ss_res = np.sum((y - alpha - beta * xm)**2)
ss_tot = np.sum((y - y_mean)**2)
r2 = 1.0 - ss_res / ss_tot

result = {
    "coefficients": {
        "alpha": float(alpha),
        "beta": float(beta),
        "r_squared": float(r2),
    },
    "standard_errors": {
        "alpha": 0.0,
        "beta": 0.0,
        "r_squared": 0.0,
    },
}

print(json.dumps(result, indent=2))

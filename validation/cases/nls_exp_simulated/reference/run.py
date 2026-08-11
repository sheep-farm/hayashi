import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
x = df["x"].values
y = df["y"].values

# y = a * exp(b * x)
def model(x, a, b):
    return a * np.exp(b * x)

popt, pcov = curve_fit(model, x, y, p0=[2.0, -1.0], method="lm")

# std errors
perr = np.sqrt(np.diag(pcov))
n = len(y)
p = 2
yhat = model(x, *popt)
resid = y - yhat
rss = np.sum(resid * resid)
sigma2 = rss / (n - p)

# Recompute using OLS standard errors with numerical Jacobian for consistency
def jacobian(params, x):
    a, b = params
    expbx = np.exp(b * x)
    ja = expbx
    jb = a * x * expbx
    return np.column_stack([ja, jb])

J = jacobian(popt, x)
JtJ = J.T @ J
cov = np.linalg.inv(JtJ) * sigma2
se = np.sqrt(np.diag(cov))

result = {
    "coefficients": {
        "a": float(popt[0]),
        "b": float(popt[1]),
    },
    "standard_errors": {
        "a": float(se[0]),
        "b": float(se[1]),
    },
}

print(json.dumps(result, indent=2))

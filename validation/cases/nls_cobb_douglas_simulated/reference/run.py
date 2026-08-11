import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
x1 = df["x1"].values
x2 = df["x2"].values
y = df["y"].values

def model(X, a, b1, b2):
    x1, x2 = X[:, 0], X[:, 1]
    return a * (x1 ** b1) * (x2 ** b2)

X = np.column_stack([x1, x2])
popt, pcov = curve_fit(model, X, y, p0=[1.0, 0.3, 0.5], method="lm")
perr = np.sqrt(np.diag(pcov))

result = {
    "coefficients": {
        "a": float(popt[0]),
        "b0": float(popt[1]),
        "b1": float(popt[2]),
    },
    "standard_errors": {
        "a": float(perr[0]),
        "b0": float(perr[1]),
        "b1": float(perr[2]),
    },
}

print(json.dumps(result, indent=2))

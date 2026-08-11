import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
x = df["x"].values
y = df["y"].values

def model(x, a, b):
    return a * (x ** b)

popt, pcov = curve_fit(model, x, y, p0=[1.5, 0.5], method="lm")
perr = np.sqrt(np.diag(pcov))

result = {
    "coefficients": {"a": float(popt[0]), "b": float(popt[1])},
    "standard_errors": {"a": float(perr[0]), "b": float(perr[1])},
}

print(json.dumps(result, indent=2))

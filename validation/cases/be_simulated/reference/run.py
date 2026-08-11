import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

gb = df.groupby("id")
means = gb[["y", "x"]].mean().reset_index()

Y = means["y"].values
X = means["x"].values
n = len(Y)

# Add intercept
Xmat = np.column_stack([np.ones(n), X])

# OLS
b = np.linalg.lstsq(Xmat, Y, rcond=None)[0]
resid = Y - Xmat @ b
s2 = np.sum(resid * resid) / (n - 2)
XX_inv = np.linalg.inv(Xmat.T @ Xmat)
se = np.sqrt(s2 * np.diag(XX_inv))

result = {
    "coefficients": {
        "x0": float(b[0]),
        "x1": float(b[1]),
    },
    "standard_errors": {
        "x0": float(se[0]),
        "x1": float(se[1]),
    },
}

print(json.dumps(result, indent=2))

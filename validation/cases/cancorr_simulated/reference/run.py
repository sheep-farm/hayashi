import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "data.csv"

df = pd.read_csv(DATA_PATH)
X = df[["x1", "x2"]].values
Y = df[["y1", "y2"]].values

# Canonical correlations from the generalised eigenvalue problem
Xc = X - X.mean(axis=0)
Yc = Y - Y.mean(axis=0)

Sxx = Xc.T @ Xc / X.shape[0]
Syy = Yc.T @ Yc / Y.shape[0]
Sxy = Xc.T @ Yc / X.shape[0]

A = np.linalg.inv(Sxx) @ Sxy @ np.linalg.inv(Syy) @ Sxy.T
eigvals = np.linalg.eigvals(A)
corrs = np.sqrt(np.sort(eigvals.real)[::-1])

wilks = float(np.prod(1 - corrs**2))

out = {
    "coefficients": {
        "cancorr_1": float(corrs[0]),
        "cancorr_2": float(corrs[1]),
        "wilks_lambda": wilks,
    },
    "standard_errors": {
        "cancorr_1": 0.0,
        "cancorr_2": 0.0,
        "wilks_lambda": 0.0,
    },
}

print(json.dumps(out, indent=2))

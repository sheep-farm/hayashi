# Reference implementation in Python for the book VAR(1) case.
# Estimates equation-by-equation OLS to match Hayashi's VAR output.

import json
from pathlib import Path

import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = VALIDATION_DIR / "data"
REF_DIR = VALIDATION_DIR / "reference"

df = pd.read_csv(DATA_DIR / "var.csv")
n = len(df)

y1 = df["y1"].to_numpy()
y2 = df["y2"].to_numpy()
y1_l1 = np.roll(y1, 1)
y2_l1 = np.roll(y2, 1)
y1_l1[0] = np.nan
y2_l1[0] = np.nan

idx = np.arange(1, n)
X = np.column_stack((np.ones(len(idx)), y1_l1[idx], y2_l1[idx]))

# Equation 1: y1
Y1 = y1[idx]
b1 = np.linalg.solve(X.T @ X, X.T @ Y1)
r1 = Y1 - X @ b1
se1 = np.sqrt(np.diag(np.sum(r1**2) / (n - 1 - 3) * np.linalg.inv(X.T @ X)))

# Equation 2: y2
Y2 = y2[idx]
b2 = np.linalg.solve(X.T @ X, X.T @ Y2)
r2 = Y2 - X @ b2
se2 = np.sqrt(np.diag(np.sum(r2**2) / (n - 1 - 3) * np.linalg.inv(X.T @ X)))

result = {
    "coefficients": {
        "y1_const": float(b1[0]),
        "y1_y1.L1": float(b1[1]),
        "y1_y2.L1": float(b1[2]),
        "y2_const": float(b2[0]),
        "y2_y1.L1": float(b2[1]),
        "y2_y2.L1": float(b2[2]),
    },
    "standard_errors": {
        "y1_const": float(se1[0]),
        "y1_y1.L1": float(se1[1]),
        "y1_y2.L1": float(se1[2]),
        "y2_const": float(se2[0]),
        "y2_y1.L1": float(se2[1]),
        "y2_y2.L1": float(se2[2]),
    },
}

REF_DIR.mkdir(parents=True, exist_ok=True)
out = json.dumps(result)
(REF_DIR / "expected.json").write_text(out)
print(out)

# Reference implementation in Python for SVAR on US macro data.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "macrodata.csv"

# Load statsmodels macro data
macro = sm.datasets.macrodata.load_pandas().data
macro = macro[["realgdp", "realcons"]].rename(columns={"realgdp": "gdp", "realcons": "cons"})
macro.to_csv(CSV_PATH, index=False)

# VAR(2) with constant
model = sm.tsa.VAR(macro)
results = model.fit(maxlags=2, trend="c")

# Cholesky identification: A = I, B = lower-triangular Cholesky of residual covariance
resid = results.resid.values
T, k = resid.shape
p = 2
Sigma = resid.T @ resid / (T - (1 + k * p))
B = np.linalg.cholesky(Sigma)
A = np.eye(2)

result = {
    "a_matrix": {f"a{i}": v for i, v in enumerate(A.flatten().tolist())},
    "b_matrix": {f"b{i}": v for i, v in enumerate(B.flatten().tolist())},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

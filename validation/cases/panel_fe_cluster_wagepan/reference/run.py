# Explicit Python reference for the Wooldridge wagepan panel FE clustered-SE case.
#
# This intentionally avoids linearmodels covariance defaults. The calculation
# mirrors within-transformed OLS with Greeners' one-way CR1 clustered covariance.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = CASE_DIR / "data" / "wagepan.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError("wagepan.csv is missing; run data/gen.py first")

variables = [
    "lwage",
    "union",
    "married",
    "d81",
    "d82",
    "d83",
    "d84",
    "d85",
    "d86",
    "d87",
    "nr",
    "year",
]
x_names = ["union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87"]

df = pd.read_csv(CSV_PATH)[variables].dropna()
clusters = df["nr"].to_numpy()

y = df["lwage"] - df.groupby("nr")["lwage"].transform("mean")
X = df[x_names] - df.groupby("nr")[x_names].transform("mean")

y_arr = y.to_numpy(dtype=float)
x_arr = X.to_numpy(dtype=float)

xtx_inv = np.linalg.inv(x_arr.T @ x_arr)
beta = xtx_inv @ x_arr.T @ y_arr
residuals = y_arr - x_arr @ beta

n, k = x_arr.shape
unique_clusters = pd.unique(clusters)
g = len(unique_clusters)

meat = np.zeros((k, k))
for cluster in unique_clusters:
    idx = clusters == cluster
    score = x_arr[idx, :].T @ residuals[idx]
    meat += np.outer(score, score)

finite_sample_correction = (g / (g - 1)) * ((n - 1) / (n - k))
vcov_cluster = finite_sample_correction * xtx_inv @ meat @ xtx_inv
se = np.sqrt(np.diag(vcov_cluster))

result = {
    "coefficients": {name: float(value) for name, value in zip(x_names, beta)},
    "standard_errors": {name: float(value) for name, value in zip(x_names, se)},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

# Reference implementation in Python for the Wooldridge card clustered-IV case.
#
# The IV clustered covariance is computed manually to match the Hayashi/Greeners
# convention and avoid package-specific finite-sample defaults.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "card.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data

        df = data("card")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/card.csv"
        df = pd.read_csv(url)
else:
    df = pd.read_csv(CSV_PATH)

region_cols = [f"reg66{i}" for i in range(1, 10)]
vars_ = [
    "lwage",
    "educ",
    "exper",
    "expersq",
    "black",
    "south",
    "smsa",
    "nearc4",
    *region_cols,
]
df = df[vars_].dropna().copy()
if not np.allclose(df[region_cols].sum(axis=1).to_numpy(dtype=float), 1.0):
    raise ValueError("card region indicators are not mutually exclusive and exhaustive")
df["region"] = np.argmax(df[region_cols].to_numpy(dtype=float), axis=1) + 1
df["feduc"] = df["educ"].astype(float)
df["fexper"] = df["exper"].astype(float)
df["fexpersq"] = df["expersq"].astype(float)
df["fblack"] = df["black"].astype(float)
df["fsouth"] = df["south"].astype(float)
df["fsmsa"] = df["smsa"].astype(float)
df["fnearc4"] = df["nearc4"].astype(float)
df.to_csv(CSV_PATH, index=False)

y = df["lwage"].to_numpy(dtype=float)[:, None]
x = np.column_stack(
    [
        np.ones(len(df), dtype=float),
        df["feduc"].to_numpy(dtype=float),
        df["fexper"].to_numpy(dtype=float),
        df["fexpersq"].to_numpy(dtype=float),
        df["fblack"].to_numpy(dtype=float),
        df["fsouth"].to_numpy(dtype=float),
        df["fsmsa"].to_numpy(dtype=float),
    ]
)
z = np.column_stack(
    [
        np.ones(len(df), dtype=float),
        df["fnearc4"].to_numpy(dtype=float),
        df["fexper"].to_numpy(dtype=float),
        df["fexpersq"].to_numpy(dtype=float),
        df["fblack"].to_numpy(dtype=float),
        df["fsouth"].to_numpy(dtype=float),
        df["fsmsa"].to_numpy(dtype=float),
    ]
)
clusters = df["region"].to_numpy()

ztz_inv = np.linalg.inv(z.T @ z)
x_hat = z @ ztz_inv @ z.T @ x
xhx_inv = np.linalg.inv(x_hat.T @ x_hat)
beta = xhx_inv @ x_hat.T @ y

resid = y - x @ beta
n, k = x.shape
cluster_levels = np.unique(clusters)
g = len(cluster_levels)

meat = np.zeros((k, k), dtype=float)
for cluster in cluster_levels:
    idx = clusters == cluster
    score = x_hat[idx, :].T @ resid[idx, :]
    meat += score @ score.T

finite_sample_correction = (g / (g - 1)) * ((n - 1) / (n - k))
vcov_cluster = finite_sample_correction * xhx_inv @ meat @ xhx_inv
variance_diag = np.diag(vcov_cluster)
if np.any(variance_diag < 0.0):
    raise ValueError("clustered IV covariance has a negative diagonal entry")
std_errors = np.sqrt(variance_diag)

names = ["Intercept", "feduc", "fexper", "fexpersq", "fblack", "fsouth", "fsmsa"]
result = {
    "coefficients": {name: float(beta[index, 0]) for index, name in enumerate(names)},
    "standard_errors": {
        name: float(std_errors[index]) for index, name in enumerate(names)
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

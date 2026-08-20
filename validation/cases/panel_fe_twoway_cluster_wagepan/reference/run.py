# Explicit Python reference for the Wooldridge wagepan panel FE two-way-clustered-SE case.
#
# Within-transformed OLS with two-way (entity + time) clustered covariance.
# Mirrors the Greeners implementation: V = sandwich(X, meat_1 + meat_2 - meat_12, X) * g/(g-1) * (n-1)/(n-k)
# where g = min(G_entity, G_time).

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

entity_name = "nr"
time_name = "year"

df = pd.read_csv(CSV_PATH)[variables].dropna()

# Within-transformation by entity.
y = df["lwage"] - df.groupby(entity_name)["lwage"].transform("mean")
X = df[x_names] - df.groupby(entity_name)[x_names].transform("mean")

y_arr = y.to_numpy(dtype=float)
x_arr = X.to_numpy(dtype=float)

xtx_inv = np.linalg.inv(x_arr.T @ x_arr)
beta = xtx_inv @ x_arr.T @ y_arr
residuals = y_arr - x_arr @ beta

n, k = x_arr.shape


def cluster_meat(cluster_col):
    clusters = df[cluster_col].to_numpy()
    unique_clusters = pd.unique(clusters)
    meat = np.zeros((k, k))
    for cluster in unique_clusters:
        idx = clusters == cluster
        x_g = x_arr[idx, :]
        u_g = residuals[idx]
        meat += x_g.T @ np.outer(u_g, u_g) @ x_g
    return meat, len(unique_clusters)


df["inter"] = df[entity_name].astype(str) + "_" + df[time_name].astype(str)
meat_1, g1 = cluster_meat(entity_name)
meat_2, g2 = cluster_meat(time_name)
meat_12, _ = cluster_meat("inter")

meat = meat_1 + meat_2 - meat_12
sandwich = xtx_inv @ meat @ xtx_inv

g = min(g1, g2)
finite_sample_correction = (g / (g - 1)) * ((n - 1) / (n - k))
vcov = finite_sample_correction * sandwich
se = np.sqrt(np.maximum(0, np.diag(vcov)))

result = {
    "coefficients": {name: float(value) for name, value in zip(x_names, beta)},
    "standard_errors": {name: float(value) for name, value in zip(x_names, se)},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

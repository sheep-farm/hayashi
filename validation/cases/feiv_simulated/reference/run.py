import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

# Within transformation by entity
entity = df["id"].values
vars = ["y", "x", "z"]
dm = df.groupby("id")[vars].transform(lambda s: s - s.mean())

Y = dm["y"].values
X = dm["x"].values
Z = dm["z"].values
n = len(Y)
G = df["id"].nunique()

# First stage: within-X on within-Z
pi = np.sum(Z * X) / np.sum(Z * Z)
Xhat = pi * Z

# Second stage: within-Y on fitted within-X
b = np.sum(Xhat * Y) / np.sum(Xhat * Xhat)

# Residuals using the observed (within) endogenous regressor
resid = Y - b * X
# Degrees of freedom account for the slope and one fixed effect per entity
s2 = np.sum(resid * resid) / (n - 1 - G)
se = np.sqrt(s2 / np.sum(Xhat * Xhat))

result = {
    "coefficients": {"x": float(b)},
    "standard_errors": {"x": float(se)},
}

print(json.dumps(result, indent=2))

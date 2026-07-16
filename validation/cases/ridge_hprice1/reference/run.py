# Reference implementation in Python for the Ridge hprice1 case.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "hprice1.csv"

df = pd.read_csv(CSV_PATH)

predictors = ["llotsize", "lsqrft", "bdrms", "colonial"]
X = df[predictors].astype(float).to_numpy()
y = df["lprice"].astype(float).to_numpy()

# Hayashi's ridge: include the intercept in the design matrix and penalise it
# together with the slope coefficients, with penalty alpha = 0.1.
X_aug = np.column_stack([np.ones(X.shape[0]), X])

n = X_aug.shape[1]
alpha = 0.1
beta = np.linalg.solve(X_aug.T @ X_aug + alpha * np.eye(n), X_aug.T @ y)

coefs = {"Intercept": float(beta[0])}
for name, val in zip(predictors, beta[1:]):
    coefs[name] = float(val)

result = {
    "coefficients": coefs,
    "standard_errors": {name: 0.0 for name in coefs},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

# Reference implementation in Python for the SUR Grunfeld case.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "grunfeld.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("grunfeld")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/grunfeld.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)


def make_design(df: pd.DataFrame, y_name: str, x_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = df[y_name].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), *[df[name].to_numpy(dtype=float) for name in x_names]])
    return y, X


# Equation 1: value ~ inv + capital
y1, X1 = make_design(df, "value", ["inv", "capital"])
# Equation 2: inv ~ value + capital
y2, X2 = make_design(df, "inv", ["value", "capital"])

n = len(df)

# Step 1: OLS per equation to get residuals.
b1 = np.linalg.solve(X1.T @ X1, X1.T @ y1)
b2 = np.linalg.solve(X2.T @ X2, X2.T @ y2)
u1 = y1 - X1 @ b1
u2 = y2 - X2 @ b2

# Step 2: Estimate Sigma.
U = np.column_stack([u1, u2])
Sigma = (U.T @ U) / n
Sigma_inv = np.linalg.inv(Sigma)

# Step 3: Build block-diagonal X and stacked y.
K1 = X1.shape[1]
K2 = X2.shape[1]
K = K1 + K2

X = np.zeros((2 * n, K))
X[:n, :K1] = X1
X[n:, K1:] = X2
y = np.concatenate([y1, y2])

# Omega_inv = Sigma_inv \otimes I_n
Omega_inv = np.zeros((2 * n, 2 * n))
Omega_inv[:n, :n] = Sigma_inv[0, 0] * np.eye(n)
Omega_inv[:n, n:] = Sigma_inv[0, 1] * np.eye(n)
Omega_inv[n:, :n] = Sigma_inv[1, 0] * np.eye(n)
Omega_inv[n:, n:] = Sigma_inv[1, 1] * np.eye(n)

# Step 4: GLS estimator.
XtOX = X.T @ Omega_inv @ X
XtOy = X.T @ Omega_inv @ y
beta = np.linalg.solve(XtOX, XtOy)

# Standard errors (classical FGLS).
vcov = np.linalg.inv(XtOX)
se = np.sqrt(np.diag(vcov))

result = {
    "coefficients": {
        "value:Intercept": float(beta[0]),
        "value:inv": float(beta[1]),
        "value:capital": float(beta[2]),
        "inv:Intercept": float(beta[3]),
        "inv:value": float(beta[4]),
        "inv:capital": float(beta[5]),
    },
    "standard_errors": {
        "value:Intercept": float(se[0]),
        "value:inv": float(se[1]),
        "value:capital": float(se[2]),
        "inv:Intercept": float(se[3]),
        "inv:value": float(se[4]),
        "inv:capital": float(se[5]),
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

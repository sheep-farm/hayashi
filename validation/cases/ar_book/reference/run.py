# Reference implementation in Python for the book AR(1) case.
# Replicates Hayashi's default Hannan-Rissanen two-step estimator.

import json
from pathlib import Path

import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = VALIDATION_DIR / "data"
REF_DIR = VALIDATION_DIR / "reference"

z = pd.read_csv(DATA_DIR / "ar.csv")["y"].to_numpy()
t = len(z)
p = 1
q = 0

# Step 1: long AR regression to get proxy residuals.
p_long = max(p + q, int(t**0.25) + 2, 4)
n_long = t - p_long

x_long = np.column_stack((np.ones(n_long), np.vstack([z[p_long - l:t - l] for l in range(1, p_long + 1)]).T))
y_long = z[p_long:]
phi_long = np.linalg.solve(x_long.T @ x_long, x_long.T @ y_long)
u_hat = y_long - x_long @ phi_long

# Step 2: regression y_t = c + ar1*z_{t-1}.
start2 = max(q, 1)
n_final = n_long - start2

x_final = np.zeros((n_final, 1 + p))
y_final = z[p_long + start2:]
for i in range(n_final):
    zi = p_long + start2 + i
    x_final[i, 0] = 1.0
    x_final[i, 1] = z[zi - 1]

beta = np.linalg.solve(x_final.T @ x_final, x_final.T @ y_final)
resid = y_final - x_final @ beta
sigma2 = np.sum(resid**2) / n_final
vc = sigma2 * np.linalg.inv(x_final.T @ x_final)
se = np.sqrt(np.diag(vc))

result = {
    "coefficients": {
        "intercept": float(beta[0]),
        "ar.L1": float(beta[1]),
    },
    "standard_errors": {
        "intercept": float(se[0]),
        "ar.L1": float(se[1]),
    },
}

REF_DIR.mkdir(parents=True, exist_ok=True)
out = json.dumps(result)
(REF_DIR / "expected.json").write_text(out)
print(out)

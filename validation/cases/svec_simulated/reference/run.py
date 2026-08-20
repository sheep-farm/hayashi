import json
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import cholesky

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
Y = df[["y1", "y2"]].values

T, k = Y.shape

# Estimate VAR(1) with intercept by OLS
Y1 = Y[1:, :]
X = np.column_stack([np.ones(T - 1), Y[:-1, :]])
B_ols = np.linalg.lstsq(X, Y1, rcond=None)[0]
resid = Y1 - X @ B_ols

# Remove intercept, keep slope matrix A1
n_obs = T - 1
n_cols_x = 1 + k
A1 = B_ols[1:, :].T
Sigma_u = (resid.T @ resid) / (n_obs - n_cols_x)

# C(1) = (I - A1 - ... - Ap)^{-1}
C1_inv = np.eye(k) - A1
C1 = np.linalg.inv(C1_inv)

# Long-run covariance = C(1) @ Sigma_u @ C(1)'
long_run_cov = C1 @ Sigma_u @ C1.T

# Lower Cholesky
try:
    lr_chol = cholesky(long_run_cov)
except np.linalg.LinAlgError:
    # add tiny jitter if needed
    lr_chol = cholesky(long_run_cov + 1e-10 * np.eye(k))

# B = C(1)^{-1} @ long_run_chol
B = C1_inv @ lr_chol
A = np.eye(k)

result = {
    "a_matrix": {"a" + str(i): v for i, v in enumerate(A.flatten(order='C').tolist())},
    "b_matrix": {"b" + str(i): v for i, v in enumerate(B.flatten(order='C').tolist())},
}

print(json.dumps(result, indent=2))

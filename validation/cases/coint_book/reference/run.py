# Reference implementation in Python for the book cointegration/VECM case.
# Replicates the Johansen ML procedure for a bivariate VECM(1) with rank 1.

import json
from pathlib import Path

import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = VALIDATION_DIR / "data"
REF_DIR = VALIDATION_DIR / "reference"


def johansen_vecm(data: np.ndarray, lags: int, rank: int) -> dict:
    """Manual Johansen ML estimation for a VECM.

    Matches the implementation in the Greeners crate (vecm.rs): data are in
    levels, the first differences are regressed on deterministic terms (here
    only an intercept) and lagged first differences, and beta/alpha are
    recovered from the canonical correlations of the residuals.
    """
    t_total = data.shape[0]
    k = data.shape[1]
    n_eff = t_total - lags
    p_vecm = lags - 1
    n_z_cols = k * p_vecm + 1

    z_mat = np.zeros((n_eff, n_z_cols))
    dy_target = np.zeros((n_eff, k))
    y_lag_level = np.zeros((n_eff, k))

    for i in range(n_eff):
        t_original = lags + i
        dy_target[i, :] = data[t_original, :] - data[t_original - 1, :]
        y_lag_level[i, :] = data[t_original - 1, :]
        z_mat[i, 0] = 1.0
        for l in range(1, p_vecm + 1):
            lag_time = t_original - l
            dy_lag = data[lag_time, :] - data[lag_time - 1, :]
            start_col = 1 + (l - 1) * k
            z_mat[i, start_col : start_col + k] = dy_lag

    ztz = z_mat.T @ z_mat
    ztz_inv = np.linalg.inv(ztz)
    beta_0 = ztz_inv @ z_mat.T @ dy_target
    r0 = dy_target - z_mat @ beta_0
    beta_1 = ztz_inv @ z_mat.T @ y_lag_level
    r1 = y_lag_level - z_mat @ beta_1

    t_float = n_eff
    s00 = r0.T @ r0 / t_float
    s11 = r1.T @ r1 / t_float
    s01 = r0.T @ r1 / t_float
    s10 = s01.T

    s11_chol = np.linalg.cholesky(s11)
    s11_inv_chol = np.linalg.inv(s11_chol)
    s00_inv = np.linalg.inv(s00)

    temp = s11_inv_chol @ s10 @ s00_inv @ s01 @ s11_inv_chol.T
    eigvals, eigvecs = np.linalg.eig(temp)

    # Keep only real eigenvalues and sort descending.
    pairs = [(val.real, eigvecs[:, i].real) for i, val in enumerate(eigvals) if np.isreal(val)]
    pairs.sort(key=lambda x: x[0], reverse=True)

    beta_est = np.zeros((k, rank))
    for r in range(rank):
        beta_vec = s11_inv_chol.T @ pairs[r][1]
        beta_est[:, r] = beta_vec

    cointegration_term = r1 @ beta_est
    alpha_est = r0.T @ cointegration_term @ np.linalg.inv(cointegration_term.T @ cointegration_term)

    # Simple OLS conditional standard errors for alpha: regress r0_j on the
    # cointegration term (already orthogonal to the constant) without intercept.
    alpha_se = np.zeros((k, rank))
    for r in range(rank):
        ec = cointegration_term[:, r]
        ss_ec = np.sum(ec * ec)
        for j in range(k):
            a = alpha_est[j, r]
            resid = r0[:, j] - a * ec
            sigma2 = np.sum(resid * resid) / max(1, n_eff - 2)
            alpha_se[j, r] = float(np.sqrt(sigma2 / ss_ec))

    # Approximate beta standard errors from the static long-run OLS regression
    # y ~ x (with intercept). The slope SE is used for beta_y2; the intercept
    # SE is used as a rough proxy for beta_y1 (the Johansen vector is not
    # normalized here, so this is intentionally approximate).
    y_level = data[:, 0]
    x_level = data[:, 1]
    X_ols = np.column_stack((np.ones(t_total), x_level))
    beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ (X_ols.T @ y_level)
    resid_ols = y_level - X_ols @ beta_ols
    sigma2_ols = np.sum(resid_ols * resid_ols) / max(1, t_total - 2)
    cov_ols = sigma2_ols * np.linalg.inv(X_ols.T @ X_ols)
    beta_se = np.zeros((k, rank))
    beta_se[0, 0] = float(np.sqrt(cov_ols[0, 0]))  # intercept SE as proxy for beta_y1
    beta_se[1, 0] = float(np.sqrt(cov_ols[1, 1]))  # slope SE as proxy for beta_y2

    return {"alpha": alpha_est, "beta": beta_est, "alpha_se": alpha_se, "beta_se": beta_se}


df = pd.read_csv(DATA_DIR / "coint.csv")
data = df[["y", "x"]].to_numpy()

res = johansen_vecm(data, lags=1, rank=1)

beta = res["beta"]
alpha = res["alpha"]
beta_se = res["beta_se"]
alpha_se = res["alpha_se"]

result = {
    "coefficients": {
        "beta_1_y1": float(beta[0, 0]),
        "beta_1_y2": float(beta[1, 0]),
        "alpha_1_y1": float(alpha[0, 0]),
        "alpha_1_y2": float(alpha[1, 0]),
    },
    "standard_errors": {
        "beta_1_y1": float(beta_se[0, 0]),
        "beta_1_y2": float(beta_se[1, 0]),
        "alpha_1_y1": float(alpha_se[0, 0]),
        "alpha_1_y2": float(alpha_se[1, 0]),
    },
}

REF_DIR.mkdir(parents=True, exist_ok=True)
out = json.dumps(result)
(REF_DIR / "expected.json").write_text(out)
print(out)

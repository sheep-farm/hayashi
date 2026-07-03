# Reference implementation in Python for RDD using the Hayashi book DGP.
#
# Data are generated once and saved to CSV; the Hayashi script reads the same
# CSV so both implementations use identical observations.

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "rdd_book.csv"

np.random.seed(42)

if not CSV_PATH.exists():
    n = 1000
    x = np.random.uniform(-1.0, 1.0, size=n)
    d = (x >= 0.0).astype(float)
    e = np.random.normal(size=n)
    y = 1.0 + 0.5 * x + 2.0 * d + e
    df = pd.DataFrame({"x": x, "y": y, "d": d})
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

x = df["x"].values
y = df["y"].values
cutoff = 0.0


def triangular_weight(u: float) -> float:
    return max(1.0 - abs(u), 0.0)


def local_poly_wls(y_s: np.ndarray, x_c: np.ndarray, w: np.ndarray, order: int):
    """Local polynomial WLS with HC1 variance for the intercept at 0."""
    n = len(y_s)
    p = order + 1
    X = np.column_stack([x_c**j for j in range(p)])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y_s
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ XtWy
    y_hat = X @ beta
    resid = y_s - y_hat
    scale = n / max(n - p, 1)
    meat = np.zeros((p, p))
    for i in range(n):
        xi = X[i]
        meat += scale * (w[i]**2) * (resid[i]**2) * np.outer(xi, xi)
    vcov = XtWX_inv @ meat @ XtWX_inv
    return beta, vcov


def side_fit(y_s: np.ndarray, x_s: np.ndarray, cutoff: float, h: float, order: int, side: str):
    mask = (x_s < cutoff) if side == "left" else (x_s >= cutoff)
    xs = x_s[mask] - cutoff
    ys = y_s[mask]
    ws = np.array([triangular_weight(u / h) for u in xs])
    valid = ws > 0
    xs, ys, ws = xs[valid], ys[valid], ws[valid]
    beta, vcov = local_poly_wls(ys, xs, ws, order)
    return beta, vcov, len(ys)


def ik_bandwidth(y_s: np.ndarray, x_s: np.ndarray, cutoff: float, order: int) -> float:
    """Imbens-Kalyanaraman bandwidth selector matching the Rust implementation."""
    n = len(x_s)
    if n < 10:
        return 1.0
    x_mean = float(np.mean(x_s))
    x_sd = float(np.std(x_s, ddof=1))
    if x_sd < 1e-15:
        return 1.0

    h0 = 1.84 * x_sd * (n**-0.2)
    q = order + 1

    def side_fit_pilot(side: str):
        mask = (x_s < cutoff) if side == "left" else (x_s >= cutoff)
        xs = x_s[mask] - cutoff
        ys = y_s[mask]
        keep = np.abs(xs) <= h0
        xs, ys = xs[keep], ys[keep]
        if len(ys) < q + 2:
            return 0.0, 1.0
        ws = np.ones(len(ys))
        beta, _ = local_poly_wls(ys, xs, ws, q)
        deriv_coeff = beta[q]
        y_hat = np.column_stack([xs**j for j in range(q + 1)]) @ beta
        resid = ys - y_hat
        resid_var = float(np.sum(resid**2) / max(len(ys) - (q + 1), 1))
        return deriv_coeff, resid_var

    m_left, sigma2_left = side_fit_pilot("left")
    m_right, sigma2_right = side_fit_pilot("right")

    b_jump = m_right - m_left
    if abs(b_jump) < 1e-12:
        return h0

    n_window = np.sum(np.abs(x_s - cutoff) <= h0)
    f_c = max(n_window / (2.0 * h0 * n), 1e-10)

    c_k = 3.4375
    exponent = 1.0 / (2.0 * order + 3.0)
    h_star = (c_k * (sigma2_left + sigma2_right) / (n * f_c * b_jump**2)) ** exponent
    return max(min(h_star, 2.0 * x_sd), 0.05 * x_sd)


# Estimate sharp RDD with local linear regression (p=1) and IK bandwidth.
order = 1
h = ik_bandwidth(y, x, cutoff, order)

beta_l, vcov_l, n_left = side_fit(y, x, cutoff, h, order, "left")
beta_r, vcov_r, n_right = side_fit(y, x, cutoff, h, order, "right")

tau = float(beta_r[0] - beta_l[0])
var_tau = max(vcov_l[0, 0] + vcov_r[0, 0], 0.0)
se = float(math.sqrt(var_tau))

result = {
    "coefficients": {"tau": tau},
    "standard_errors": {"tau": se},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

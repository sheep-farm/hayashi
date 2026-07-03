# Reference implementation in Python for the Heckman two-step (Heckit) case.
#
# Uses a manual two-step estimator because pyheckit is not assumed to be installed.

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "mroz.csv"

# Variables required for this case.
SELECTION_VARS = ["educ", "age", "kidslt6", "kidsge6", "nwifeinc"]
OUTCOME_VARS = ["educ", "exper", "expersq"]

NEEDED_COLUMNS = ["inlf", "lwage", "educ", "age", "kidslt6", "kidsge6", "nwifeinc", "exper", "expersq"]

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        full_df = data("mroz")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/mroz.csv"
        full_df = pd.read_csv(url)
    # Keep only the columns required for this case and make lwage numeric for Hayashi.
    df = full_df[NEEDED_COLUMNS].copy()
    df["lwage"] = df["lwage"].fillna(0.0)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)


# ---------------------------------------------------------------------------
# Manual two-step Heckman (Heckit)
# ---------------------------------------------------------------------------

def inverse_mills_ratio(z_hat: np.ndarray) -> np.ndarray:
    """Compute the inverse Mills ratio with a tail approximation."""
    phi = norm.pdf(z_hat)
    cdf = np.clip(norm.cdf(z_hat), 1e-300, 1.0)
    lam = phi / cdf
    lam = np.where(z_hat < -30.0, -z_hat - 1.0 / z_hat, lam)
    return lam


def delta_i(z_hat: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Compute δ_i = λ_i (λ_i + z_hat_i) with a tail approximation."""
    d = lam * (lam + z_hat)
    d = np.where(z_hat < -30.0, 1.0 + 1.0 / (z_hat * z_hat), d)
    return d


def fit_probit_newton(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probit MLE via Newton-Raphson, returning coefficients and V_gamma."""
    beta = np.zeros(x.shape[1])
    for _ in range(200):
        xb = x @ beta
        p = np.clip(norm.cdf(xb), 1e-15, 1.0 - 1e-15)
        phi = norm.pdf(xb)
        w = phi / (p * (1.0 - p))
        score = x.T @ ((y - p) * w)
        hess = -x.T @ (x * ((w * phi)[:, np.newaxis]))
        step = np.linalg.solve(hess, score)
        beta = beta - step
        if np.linalg.norm(step) < 1e-7:
            break
    xb = x @ beta
    p = np.clip(norm.cdf(xb), 1e-15, 1.0 - 1e-15)
    phi = norm.pdf(xb)
    w = phi / (p * (1.0 - p))
    hess = -x.T @ (x * ((w * phi)[:, np.newaxis]))
    return beta, np.linalg.inv(-hess)


def heckit_two_step(y, x_out, z, x_sel, gamma, v_gamma):
    """Manual two-step Heckman with Heckman (1979) corrected SEs.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable (selected observations only).
    x_out : np.ndarray
        Outcome regressors including intercept (selected observations only).
    z : np.ndarray
        Selection indicator (0/1) for all observations.
    x_sel : np.ndarray
        Selection regressors including intercept for all observations.
    gamma : np.ndarray
        Probit coefficients from the selection equation.
    v_gamma : np.ndarray
        Covariance matrix of the probit coefficients.

    Returns
    -------
    beta, beta_se, delta_hat, delta_se, sigma
    """
    n1 = x_out.shape[0]
    k1 = x_out.shape[1]

    z_hat = x_sel @ gamma
    lam = inverse_mills_ratio(z_hat)
    d_i = delta_i(z_hat, lam)

    # OLS of outcome on covariates plus IMR.
    w = np.column_stack((x_out, lam[z == 1.0]))
    beta_delta = np.linalg.solve(w.T @ w, w.T @ y)
    beta = beta_delta[:k1]
    delta_hat = beta_delta[k1]

    resid = y - w @ beta_delta
    ssr = resid @ resid
    sigma2 = (ssr + delta_hat**2 * d_i[z == 1.0].sum()) / n1
    sigma = math.sqrt(sigma2)

    wtw_inv = np.linalg.inv(w.T @ w)
    x_sel_1 = x_sel[z == 1.0, :]
    d = np.diag(d_i[z == 1.0])
    xtd_xs = w.T @ d @ x_sel_1
    correction_meat = xtd_xs @ v_gamma @ xtd_xs.T
    correction = wtw_inv @ correction_meat @ wtw_inv * (delta_hat**2)
    vcov = wtw_inv * sigma2 + correction

    beta_se = np.sqrt(np.maximum(np.diag(vcov[:k1, :k1]), 0.0))
    delta_se = math.sqrt(max(vcov[k1, k1], 0.0))

    return beta, beta_se, delta_hat, delta_se, sigma


# Use statsmodels if available; otherwise fall back to the internal Newton-Raphson.
try:
    import statsmodels.api as sm
    x_sel = np.column_stack((np.ones(len(df)), df[SELECTION_VARS].to_numpy(dtype=float)))
    z = df["inlf"].to_numpy(dtype=float)
    probit = sm.Probit(z, x_sel).fit(disp=0)
    gamma = np.asarray(probit.params)
    v_gamma = np.asarray(probit.cov_params())
except Exception:
    x_sel = np.column_stack((np.ones(len(df)), df[SELECTION_VARS].to_numpy(dtype=float)))
    z = df["inlf"].to_numpy(dtype=float)
    gamma, v_gamma = fit_probit_newton(z, x_sel)

selected = z == 1.0
x_out_all = np.column_stack((np.ones(len(df)), df[OUTCOME_VARS].to_numpy(dtype=float)))
x_out = x_out_all[selected, :]
y = df.loc[selected, "lwage"].to_numpy(dtype=float)

beta, beta_se, delta_hat, delta_se, sigma = heckit_two_step(
    y, x_out, z, x_sel, gamma, v_gamma
)

result = {
    "coefficients": {
        "Intercept": float(beta[0]),
        "educ": float(beta[1]),
        "exper": float(beta[2]),
        "expersq": float(beta[3]),
        "lambda_IMR": float(delta_hat),
    },
    "standard_errors": {
        "Intercept": float(beta_se[0]),
        "educ": float(beta_se[1]),
        "exper": float(beta_se[2]),
        "expersq": float(beta_se[3]),
        "lambda_IMR": float(delta_se),
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

# Reference implementation in Python for the GARCH NYSE returns case.

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
REF_DIR = CASE_DIR / "reference"

df = pd.read_csv(DATA_DIR / "nyse.csv")
returns = df["return"].astype(float).dropna().to_numpy()


def _garch11_mle(r):
    """GARCH(1,1) MLE with positive parameters and alpha+beta < 1."""
    T = r.size
    rmean = r.mean()
    rvar = r.var(ddof=0)

    def _garch_ll(theta):
        # theta = [mu, log_omega, alpha_raw, beta_raw]
        mu = theta[0]
        omega = np.exp(theta[1])
        ex = np.exp(theta[2:4])
        denom = 1.0 + ex.sum()
        alpha, beta = ex / denom
        if alpha + beta >= 0.9999:
            return 1e12

        h = np.empty(T)
        # long-run variance as initial value
        h[0] = omega / max(1e-12, 1.0 - alpha - beta)
        eps2 = (r - mu) ** 2
        for t in range(1, T):
            h[t] = omega + alpha * eps2[t - 1] + beta * h[t - 1]
        h = np.maximum(h, 1e-12)
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(h) + eps2 / h)
        return -ll.sum()

    init = np.array([rmean, np.log(0.1 * rvar), -2.0, 2.0])
    res = minimize(_garch_ll, init, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8})
    if not res.success:
        res = minimize(_garch_ll, res.x, method="Powell", options={"maxiter": 5000})

    mu, log_omega, a, b = res.x
    omega = np.exp(log_omega)
    ex = np.exp(np.array([a, b]))
    denom = 1.0 + ex.sum()
    alpha, beta = ex / denom

    # Numerical Hessian for standard errors
    def _free_ll(theta):
        return _garch_ll(theta)

    eps = 1e-5
    n = 4
    H = np.zeros((n, n))
    x = res.x.copy()
    for i in range(n):
        for j in range(n):
            x_ij = x.copy()
            x_i = x.copy()
            x_j = x.copy()
            x_ij[i] += eps
            x_ij[j] += eps
            x_i[i] += eps
            x_j[j] += eps
            H[i, j] = (_free_ll(x_ij) - _free_ll(x_i) - _free_ll(x_j) + _free_ll(x)) / (eps ** 2)
    try:
        cov = np.linalg.inv(H + 1e-6 * np.eye(n))
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        se = np.zeros(n)

    return {
        "mu": float(mu),
        "omega": float(omega),
        "alpha[1]": float(alpha),
        "beta[1]": float(beta),
    }, {
        "mu": float(se[0]),
        "omega": float(se[1]),
        "alpha[1]": float(se[2]),
        "beta[1]": float(se[3]),
    }


coefs, std_errors = _garch11_mle(returns)

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

REF_DIR.mkdir(parents=True, exist_ok=True)
with open(REF_DIR / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

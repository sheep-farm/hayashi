# Reference implementation in Python for the book GARCH(1,1) case.

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

VALIDATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = VALIDATION_DIR / "data"
REF_DIR = VALIDATION_DIR / "reference"

df = pd.read_csv(DATA_DIR / "garch.csv")

def _garch11_mle(r):
    """GARCH(1,1) MLE with positive parameters and alpha+beta < 1."""
    T = r.size
    rmean = r.mean()
    rvar = r.var(ddof=0)

    def _garch_ll(theta):
        mu = theta[0]
        omega = np.exp(theta[1])
        ex = np.exp(theta[2:4])
        denom = 1.0 + ex.sum()
        alpha, beta = ex / denom
        if alpha + beta >= 0.9999:
            return 1e12

        h = np.empty(T)
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
    return {
        "mu": float(mu),
        "omega": float(omega),
        "alpha[1]": float(alpha),
        "beta[1]": float(beta),
    }


coefs = _garch11_mle(df["e"].astype(float).dropna().to_numpy())

result = {"coefficients": coefs}

REF_DIR.mkdir(parents=True, exist_ok=True)
out = json.dumps(result)
(REF_DIR / "expected.json").write_text(out)
print(out)

# Reference implementation in Python for the EGARCH NYSE returns case.

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


def _egarch11_mle(r):
    """EGARCH(1,1) MLE: log(h_t) = omega + alpha*|z_{t-1}| + gamma*z_{t-1} + beta*log(h_{t-1})."""
    T = r.size
    rmean = r.mean()
    rvar = r.var(ddof=0)

    def _egarch_ll(theta):
        # [mu, omega, alpha, gamma, beta_raw]
        mu = theta[0]
        omega = theta[1]
        alpha = theta[2]
        gamma = theta[3]
        beta = np.tanh(theta[4])
        if abs(beta) >= 0.9999:
            return 1e12

        log_h = np.empty(T)
        log_h[0] = np.log(rvar)
        eps = r - mu
        for t in range(1, T):
            z = eps[t - 1] / np.exp(0.5 * log_h[t - 1])
            log_h[t] = omega + alpha * abs(z) + gamma * z + beta * log_h[t - 1]
        h = np.maximum(np.exp(log_h), 1e-12)
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(h) + eps ** 2 / h)
        return -ll.sum()

    init = np.array([rmean, -0.3, 0.2, -0.1, np.arctanh(0.5)])
    res = minimize(_egarch_ll, init, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8})
    if not res.success:
        res = minimize(_egarch_ll, res.x, method="Powell", options={"maxiter": 5000})

    mu, omega, alpha, gamma, beta_raw = res.x
    beta = float(np.tanh(beta_raw))

    # Numerical Hessian for standard errors
    def _free_ll(theta):
        return _egarch_ll(theta)

    eps = 1e-5
    n = 5
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
        "gamma[1]": float(gamma),
        "beta[1]": float(beta),
        "alpha[1]": float(alpha),
    }, {
        "mu": float(se[0]),
        "omega": float(se[1]),
        "gamma[1]": float(se[2]),
        "beta[1]": float(se[3]),
        "alpha[1]": float(se[4]),
    }


coefs, std_errors = _egarch11_mle(returns)

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

REF_DIR.mkdir(parents=True, exist_ok=True)
with open(REF_DIR / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

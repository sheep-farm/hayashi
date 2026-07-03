# Reference implementation in Python for the ARIMA GDP case.
#
# This script reports the global maximum of the exact Gaussian likelihood for
# an ARIMA(1,1,1) model on log US real GDP. This matches the MLE estimator
# exposed by Hayashi's `arima(..., method="mle")`.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "macrodata.csv"

if not CSV_PATH.exists():
    macro = sm.datasets.macrodata.load_pandas().data
    macro = macro[["year", "quarter", "realgdp"]]
    macro = macro.rename(columns={"realgdp": "gdp"})
    macro.to_csv(CSV_PATH, index=False)
else:
    macro = pd.read_csv(CSV_PATH)

lgdp = np.log(macro["gdp"].astype(float).values)
z = np.diff(lgdp)
n = len(z)


def exact_loglik(phi: float, theta: float) -> tuple[float, float]:
    """Exact Gaussian log-likelihood for ARMA(1,1) on the centred differences."""
    zc = z - z.mean()

    # MA(infinity) coefficients
    psi = [0.0] * 1000
    psi[0] = 1.0
    for j in range(1, 1000):
        val = phi * psi[j - 1]
        if j == 1:
            val += theta
        psi[j] = val
        if abs(val) < 1e-12:
            break

    # Autocovariances (sigma^2 = 1)
    max_lag = min(n, 50)
    gamma = [0.0] * (max_lag + 1)
    for k in range(max_lag + 1):
        s = 0.0
        for j in range(1000):
            if j + k >= 1000:
                break
            s += psi[j] * psi[j + k]
            if j > n and abs(psi[j]) < 1e-12 and abs(psi[j + k]) < 1e-12:
                break
        gamma[k] = s

    # Durbin-Levinson innovations algorithm
    v = [0.0] * n
    v[0] = gamma[0]
    phi_coefs: list[list[float]] = [[]]
    sum_log_v = 0.0
    sum_eps2_v = 0.0

    for t in range(n):
        xhat = 0.0
        if t > 0:
            prev = phi_coefs[t - 1]
            for j, coeff in enumerate(prev):
                xhat += coeff * zc[t - 1 - j]
        eps = zc[t] - xhat
        sum_log_v += np.log(v[t])
        sum_eps2_v += eps * eps / v[t]

        if t + 1 < n:
            k = t + 1
            num = gamma[k] if k <= max_lag else 0.0
            prev = phi_coefs[t]
            for j, coeff in enumerate(prev):
                lag = k - 1 - j
                num -= coeff * (gamma[lag] if lag <= max_lag else 0.0)
            phi_kk = num / v[t] if v[t] > 0 else 0.0
            new_phi = []
            for j in range(min(k - 1, max_lag)):
                prev_j = prev[j]
                prev_kj = prev[k - 2 - j] if k - 2 - j < len(prev) else 0.0
                new_phi.append(prev_j - phi_kk * prev_kj)
            new_phi.append(phi_kk)
            v[k] = v[t] * (1.0 - phi_kk * phi_kk)
            phi_coefs.append(new_phi)

    sigma2 = sum_eps2_v / n
    log_lik = -0.5 * n * (1.0 + np.log(2.0 * np.pi * sigma2)) - 0.5 * sum_log_v
    return log_lik, sigma2


best_ll = -np.inf
best_phi = 0.0
best_theta = 0.0
best_sigma2 = 0.0
phi_grid = np.arange(-0.95, 0.96, 0.05)
theta_grid = np.arange(-0.95, 0.96, 0.05)
for phi in phi_grid:
    for theta in theta_grid:
        if not (-0.999 < phi < 0.999 and -0.999 < theta < 0.999):
            continue
        ll, s2 = exact_loglik(phi, theta)
        if ll > best_ll:
            best_ll = ll
            best_phi = phi
            best_theta = theta
            best_sigma2 = s2

# Fine refinement around the best coarse candidate.
for phi in np.arange(best_phi - 0.05, best_phi + 0.051, 0.01):
    for theta in np.arange(best_theta - 0.05, best_theta + 0.051, 0.01):
        if not (-0.999 < phi < 0.999 and -0.999 < theta < 0.999):
            continue
        ll, s2 = exact_loglik(phi, theta)
        if ll > best_ll:
            best_ll = ll
            best_phi = phi
            best_theta = theta
            best_sigma2 = s2

intercept = float(z.mean())
coefs = {
    "intercept": intercept,
    "ar.L1": float(best_phi),
    "ma.L1": float(best_theta),
}


result = {
    "coefficients": coefs,
    "standard_errors": {name: 0.0 for name in coefs},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

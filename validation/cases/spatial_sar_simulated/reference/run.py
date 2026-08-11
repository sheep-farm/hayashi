import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load data and W matrix
df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
W = np.array(pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "W.csv", header=None))

y = df["y"].values
x = df["x"].values
n = len(y)

# Design matrix with intercept
X = np.column_stack([np.ones(n), x])

# Eigenvalues of W for exact log-det
w_eigvals = np.linalg.eigvals(W).real


def sar_loglik(rho):
    A = np.eye(n) - rho * W
    y_star = y - rho * W @ y
    beta = np.linalg.lstsq(X, y_star, rcond=None)[0]
    resid = y_star - X @ beta
    rss = np.sum(resid ** 2)
    sigma2 = rss / n
    log_det = np.sum(np.log(np.abs(1 - rho * w_eigvals)))
    ll = log_det - n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return ll


# Grid search over rho
rhos = np.linspace(-0.99, 0.99, 199)
lls = np.array([sar_loglik(r) for r in rhos])
best_rho = rhos[np.argmax(lls)]

# Golden-section refinement
lo, hi = best_rho - 0.05, best_rho + 0.05
phi = 0.6180339887498949
a, b = lo, hi
c = b - phi * (b - a)
d = a + phi * (b - a)
fc = sar_loglik(c)
fd = sar_loglik(d)
for _ in range(50):
    if fc > fd:
        b, d, fd = d, c, fc
        c = b - phi * (b - a)
        fc = sar_loglik(c)
    else:
        a, c, fc = c, d, fc
        d = a + phi * (b - a)
        fd = sar_loglik(d)
best_rho = c if fc > fd else d

# Final beta and SEs
wy = W @ y
y_star = y - best_rho * wy
beta = np.linalg.lstsq(X, y_star, rcond=None)[0]

# Compute standard errors using transformed residuals
fitted = X @ beta + best_rho * wy
resid = y - fitted
sigma2 = np.sum(resid ** 2) / n

xtx_inv = np.linalg.inv(X.T @ X)
beta_se = np.sqrt(np.diag(xtx_inv * sigma2))

result = {
    "coefficients": {
        "rho (spatial lag)": float(best_rho),
        "_cons": float(beta[0]),
        "x": float(beta[1]),
    },
    "standard_errors": {
        "rho": float("nan"),  # not easily recovered from this MLE
        "_cons": float(beta_se[0]),
        "x": float(beta_se[1]),
    },
}

print(json.dumps(result, indent=2))

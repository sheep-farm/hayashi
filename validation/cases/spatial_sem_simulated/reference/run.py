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


def sem_loglik(lam):
    B = np.eye(n) - lam * W
    # Filtered y and X
    y_tilde = B @ y
    X_tilde = B @ X
    beta = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]
    resid = y_tilde - X_tilde @ beta
    rss = np.sum(resid ** 2)
    sigma2 = rss / n
    log_det = np.sum(np.log(np.abs(1 - lam * w_eigvals)))
    ll = log_det - n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return ll


# Grid search over lambda
lams = np.linspace(-0.99, 0.99, 199)
lls = np.array([sem_loglik(l) for l in lams])
best_lam = lams[np.argmax(lls)]

# Golden-section refinement
lo, hi = best_lam - 0.05, best_lam + 0.05
phi = 0.6180339887498949
a, b = lo, hi
c = b - phi * (b - a)
d = a + phi * (b - a)
fc = sem_loglik(c)
fd = sem_loglik(d)
for _ in range(50):
    if fc > fd:
        b, d, fd = d, c, fc
        c = b - phi * (b - a)
        fc = sem_loglik(c)
    else:
        a, c, fc = c, d, fc
        d = a + phi * (b - a)
        fd = sem_loglik(d)
best_lam = c if fc > fd else d

# Final beta and SEs
B = np.eye(n) - best_lam * W
y_tilde = B @ y
X_tilde = B @ X
beta = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]

resid = y_tilde - X_tilde @ beta
sigma2 = np.sum(resid ** 2) / n

xtx_inv = np.linalg.inv(X_tilde.T @ X_tilde)
beta_se = np.sqrt(np.diag(xtx_inv * sigma2))

result = {
    "coefficients": {
        "lambda": float(best_lam),
        "_cons": float(beta[0]),
        "x": float(beta[1]),
    },
    "standard_errors": {
        "lambda": 0.0,
        "_cons": float(beta_se[0]),
        "x": float(beta_se[1]),
    },
}

print(json.dumps(result, indent=2))

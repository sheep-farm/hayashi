import json
import numpy as np
import pandas as pd
from pathlib import Path

# Spatial Durbin concentrated MLE for the spatial autoregressive parameter
# Reference optimises the log-likelihood over rho for y = rho*W*y + X*beta + W*X*theta + e

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
W = np.array(pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "W.csv", header=None))

y = df["y"].values
x = df["x"].values
n = len(y)

# Standardized (but not row-normalised) rook W from the same 7x7 grid
# Reference uses the provided W.csv directly.

# Eigenvalues of W
w_eigvals = np.linalg.eigvals(W).real

# Design: [X, WX] without duplicate intercept (W is row-stochastic)
X = np.column_stack([np.ones(n), x])
Wx = (W @ x).reshape(-1, 1)
Wy = W @ y
Z = np.column_stack([X, Wx])


def durbin_loglik(rho):
    A = np.eye(n) - rho * W
    y_star = y - rho * Wy
    beta = np.linalg.lstsq(Z, y_star, rcond=None)[0]
    resid = y_star - Z @ beta
    rss = np.sum(resid ** 2)
    sigma2 = rss / n
    log_det = np.sum(np.log(np.abs(1 - rho * w_eigvals)))
    ll = log_det - n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return ll


# Grid search over rho
rhos = np.linspace(-0.99, 0.99, 199)
lls = np.array([durbin_loglik(r) for r in rhos])
best_rho = rhos[np.argmax(lls)]

# Golden-section refinement
lo, hi = best_rho - 0.05, best_rho + 0.05
phi = 0.6180339887498949
a, b = lo, hi
c = b - phi * (b - a)
d = a + phi * (b - a)
fc = durbin_loglik(c)
fd = durbin_loglik(d)
for _ in range(50):
    if fc > fd:
        b, d, fd = d, c, fc
        c = b - phi * (b - a)
        fc = durbin_loglik(c)
    else:
        a, c, fc = c, d, fc
        d = a + phi * (b - a)
        fd = durbin_loglik(d)
best_rho = c if fc > fd else d

result = {
    "coefficients": {
        "rho": float(best_rho),
    },
    "standard_errors": {
        "rho": float("nan"),
    },
}

print(json.dumps(result, indent=2))

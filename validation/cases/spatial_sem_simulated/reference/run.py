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


def sem_loglik(lambda_):
    A = np.eye(n) - lambda_ * W
    u = A.dot(y) - A.dot(X).dot(np.linalg.lstsq(A.dot(X), A.dot(y), rcond=None)[0])
    rss = np.sum(u ** 2)
    sigma2 = rss / n
    log_det = np.sum(np.log(np.abs(1 - lambda_ * w_eigvals)))
    ll = log_det - n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return ll


# Grid search over lambda
lambdas = np.linspace(-0.99, 0.99, 199)
lls = np.array([sem_loglik(l) for l in lambdas])
best_lambda = lambdas[np.argmax(lls)]

# Golden-section refinement
lo, hi = best_lambda - 0.05, best_lambda + 0.05
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
best_lambda = c if fc > fd else d

# Final beta
A = np.eye(n) - best_lambda * W
AX = A.dot(X)
Ay = A.dot(y)
beta = np.linalg.lstsq(AX, Ay, rcond=None)[0]

result = {
    "coefficients": {
        "lambda (spatial error)": float(best_lambda),
        "_cons": float(beta[0]),
        "x": float(beta[1]),
    },
}

print(json.dumps(result, indent=2))

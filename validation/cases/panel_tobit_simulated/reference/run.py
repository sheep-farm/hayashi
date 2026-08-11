import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
x = df["x"].values
y = df["y"].values
n = len(y)

X = np.column_stack([np.ones(n), x])

def neg_loglik(params):
    beta = params[:2]
    sigma = np.exp(params[2])
    xb = X @ beta
    uncens = y > 0
    ll = np.zeros(n)
    ll[uncens] = norm.logpdf((y[uncens] - xb[uncens]) / sigma) - np.log(sigma)
    ll[~uncens] = norm.logcdf((0 - xb[~uncens]) / sigma)
    return -np.sum(ll)

res = minimize(neg_loglik, [1.0, 0.5, 0.0], method="BFGS")
beta = res.x[:2]

# SE by numerical Hessian
eps = 1e-5
H = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        p1 = res.x.copy()
        p2 = res.x.copy()
        p3 = res.x.copy()
        p4 = res.x.copy()
        p1[i] += eps; p1[j] += eps
        p2[i] += eps; p2[j] -= eps
        p3[i] -= eps; p3[j] += eps
        p4[i] -= eps; p4[j] -= eps
        H[i, j] = (neg_loglik(p1) - neg_loglik(p2) - neg_loglik(p3) + neg_loglik(p4)) / (4 * eps ** 2)

se = np.sqrt(np.diag(np.linalg.inv(H)))[:2]

result = {
    "coefficients": {"_cons": float(beta[0]), "x": float(beta[1])},
    "standard_errors": {"_cons": float(se[0]), "x": float(se[1])},
}

print(json.dumps(result, indent=2))

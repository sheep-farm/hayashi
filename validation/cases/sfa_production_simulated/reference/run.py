import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["ly"].values
X = np.column_stack([np.ones(len(df)), df["lx1"].values, df["lx2"].values])
n = len(y)


def negative_log_likelihood(params):
    beta = params[:3]
    log_sv = params[3]
    log_su = params[4]
    sv = np.exp(log_sv)
    su = np.exp(log_su)
    eps = y - X @ beta
    z = -eps * su / (sv ** 2)
    ll = (
        n * np.log(2.0)
        - n * np.log(sv)
        - 0.5 * np.sum((eps / sv) ** 2)
        + np.sum(norm.logcdf(z))
    )
    return -ll


start_beta = np.linalg.lstsq(X, y, rcond=None)[0]
start = np.concatenate([start_beta, [np.log(0.15), np.log(0.1)]])

res = minimize(
    negative_log_likelihood,
    start,
    method="L-BFGS-B",
    bounds=[(None, None), (None, None), (None, None), (-6.0, 2.0), (-15.0, 2.0)],
)

beta = res.x[:3]
resid = y - X @ beta
sigma2 = np.sum(resid ** 2) / (n - 3)
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * sigma2))

result = {
    "coefficients": {
        "const": float(beta[0]),
        "lx1": float(beta[1]),
        "lx2": float(beta[2]),
    },
    "standard_errors": {
        "const": float(se[0]),
        "lx1": float(se[1]),
        "lx2": float(se[2]),
    },
}

print(json.dumps(result, indent=2))

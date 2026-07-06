# Reference implementation in Python for the Tobit hours-worked case.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "mroz.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("mroz")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/mroz.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Tobit: hours censored at zero.
vars_ = ["hours", "nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]
df = df[vars_].dropna()

y = df["hours"].to_numpy(dtype=float)
X = df[["nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]].to_numpy(dtype=float)
X = np.column_stack([np.ones(len(y)), X])


def tobit_ll(params):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    xb = X @ beta
    uncensored = y > 0
    ll = np.empty(len(y))
    z = (y[uncensored] - xb[uncensored]) / sigma
    ll[uncensored] = np.log(norm.pdf(z)) - np.log(sigma)
    z0 = -xb[~uncensored] / sigma
    ll[~uncensored] = np.log(norm.cdf(z0))
    return -ll.sum()


init_beta = np.zeros(X.shape[1])
init_beta[0] = y.mean()
init_params = np.concatenate([init_beta, [np.log(y.std())]])

result = minimize(tobit_ll, init_params, method="BFGS")

beta = result.x[:-1]

# Numerical Hessian for standard errors.
from statsmodels.tools.numdiff import approx_hess

hess = approx_hess(result.x, tobit_ll)
cov = np.linalg.inv(hess + 1e-8 * np.eye(len(hess)))
se = np.sqrt(np.diag(cov))

names = ["const", "nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]
coefs = {name: float(val) for name, val in zip(names, beta)}
std_errors = {name: float(val) for name, val in zip(names, se[:-1])}

result_dict = {
    "coefficients": coefs,
    "standard_errors": std_errors,
    "diagnostics": {
        "success": bool(result.success),
        "message": str(result.message),
        "log_likelihood": float(-result.fun),
        "sigma": float(np.exp(result.x[-1])),
        "nobs": int(len(y)),
        "censored": int((y <= 0).sum()),
        "uncensored": int((y > 0).sum()),
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result_dict, f, indent=2)

print(json.dumps(result_dict))

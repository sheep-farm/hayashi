#!/usr/bin/env python3
"""Python reference for the Tobit hours-worked case.

Left-censored Tobit MLE implemented manually and refined with Nelder-Mead
after an initial BFGS pass. Standard errors come from the numerical Hessian.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess

CASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = CASE_DIR / "data" / "mroz.csv"

df = pd.read_csv(CSV_PATH)
vars_ = ["hours", "nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]
df = df[vars_].dropna()

y = df["hours"].to_numpy(float)
X = df[["nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]].to_numpy(float)
X = np.column_stack([np.ones(len(y)), X])

cens = y <= 0
uncens = y > 0
n = len(y)


def nll(params):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    xb = X @ beta

    ll = np.empty(n)
    z_u = (y[uncens] - xb[uncens]) / sigma
    ll[uncens] = norm.logpdf(z_u) - np.log(sigma)
    z_c = -xb[cens] / sigma
    ll[cens] = norm.logcdf(z_c)

    return -ll.sum()


# Initial BFGS pass, then a short Nelder-Mead polish to improve precision.
init = np.zeros(X.shape[1])
init[0] = y.mean()
init = np.concatenate([init, [np.log(y.std())]])

res = minimize(nll, init, method="BFGS")
res = minimize(
    nll,
    res.x,
    method="Nelder-Mead",
    options={
        "xatol": 1e-12,
        "fatol": 1e-12,
        "maxiter": 20000,
        "adaptive": True,
    },
)

beta = res.x[:-1]
log_sigma = res.x[-1]
sigma = float(np.exp(log_sigma))

hess = approx_hess(res.x, nll)
cov = np.linalg.inv(hess + 1e-8 * np.eye(len(hess)))
se = np.sqrt(np.diag(cov))

names = ["const", "nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]

coefs = {name: float(val) for name, val in zip(names, beta)}
std_errors = {name: float(val) for name, val in zip(names, se[:-1])}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
    "diagnostics": {
        "log_likelihood": float(-res.fun),
        "sigma": sigma,
        "nobs": int(n),
        "censored": int(cens.sum()),
        "uncensored": int(uncens.sum()),
    },
}

print(json.dumps(result, indent=2))

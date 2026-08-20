# Reference implementation in Python/statsmodels for the Wooldridge phillips
# Newey-West OLS case.
#
# The HAC covariance is computed manually to make the kernel, lag length, and
# finite-sample correction match the Hayashi/Greeners convention.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "phillips.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data

        df = data("phillips")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/phillips.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=["cinf", "unem"])
model = smf.ols("cinf ~ 1 + unem", data=df).fit()


def newey_west_vcov(fitted_model, lags: int) -> np.ndarray:
    x = np.asarray(fitted_model.model.exog, dtype=float)
    residuals = np.asarray(fitted_model.resid, dtype=float)
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)

    meat = x.T @ (x * (residuals**2)[:, None])

    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        omega_l = np.zeros((k, k), dtype=float)

        for t in range(lag, n):
            scale = residuals[t] * residuals[t - lag]
            omega_l += scale * np.outer(x[t, :], x[t - lag, :])

        meat += weight * (omega_l + omega_l.T)

    correction = n / (n - k)
    return correction * xtx_inv @ meat @ xtx_inv


vcov_nw = newey_west_vcov(model, lags=4)
std_errors = np.sqrt(np.maximum(np.diag(vcov_nw), 0.0))

coefs = {name: float(value) for name, value in model.params.items()}
standard_errors = {
    name: float(std_errors[index])
    for index, name in enumerate(model.model.exog_names)
}

result = {
    "coefficients": coefs,
    "standard_errors": standard_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

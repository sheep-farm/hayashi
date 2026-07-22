# Reference implementation in Python for the Wooldridge mroz robust-IV case.
#
# The IV HC1 covariance is computed manually to match the Hayashi/Greeners
# convention and avoid package-specific finite-sample defaults.

import json
from pathlib import Path

import numpy as np
import pandas as pd

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

vars_ = ["lwage", "educ", "exper", "expersq", "fatheduc", "motheduc"]
df = df[vars_].dropna()

y = df["lwage"].to_numpy(dtype=float)[:, None]
x = np.column_stack(
    [
        np.ones(len(df), dtype=float),
        df["educ"].to_numpy(dtype=float),
        df["exper"].to_numpy(dtype=float),
        df["expersq"].to_numpy(dtype=float),
    ]
)
z = np.column_stack(
    [
        np.ones(len(df), dtype=float),
        df["fatheduc"].to_numpy(dtype=float),
        df["motheduc"].to_numpy(dtype=float),
        df["exper"].to_numpy(dtype=float),
        df["expersq"].to_numpy(dtype=float),
    ]
)

ztz_inv = np.linalg.inv(z.T @ z)
x_hat = z @ ztz_inv @ z.T @ x
xhx_inv = np.linalg.inv(x_hat.T @ x_hat)
beta = xhx_inv @ x_hat.T @ y

resid = y - x @ beta
n, k = x.shape

meat = x_hat.T @ (x_hat * (resid[:, 0] ** 2)[:, None])
vcov_hc1 = (n / (n - k)) * xhx_inv @ meat @ xhx_inv
std_errors = np.sqrt(np.maximum(np.diag(vcov_hc1), 0.0))

names = ["Intercept", "educ", "exper", "expersq"]
result = {
    "coefficients": {name: float(beta[index, 0]) for index, name in enumerate(names)},
    "standard_errors": {
        name: float(std_errors[index]) for index, name in enumerate(names)
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

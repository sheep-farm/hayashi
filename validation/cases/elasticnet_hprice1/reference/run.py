# Reference implementation in Python for the Elastic Net hprice1 case.
#
# Replicates Hayashi's elasticnet() coordinate descent with alpha=0.1 and
# l1_ratio=0.5.  y and X are centered and standardized, the intercept is
# unpenalised, and the penalty is alpha * n_obs * (l1_ratio*|beta|_1 +
# (1-l1_ratio)/2 * ||beta||_2^2).

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "hprice1.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("hprice1")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/hprice1.csv"
        df = pd.read_csv(url)
    df["lprice"] = np.log(df["price"])
    df["llotsize"] = np.log(df["lotsize"])
    df["lsqrft"] = np.log(df["sqrft"])
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

predictors = ["llotsize", "lsqrft", "bdrms", "colonial"]
X = df[predictors].astype(float).to_numpy()
y = df["lprice"].astype(float).to_numpy()


def _elasticnet_cd(x, y, alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-6):
    """Coordinate descent matching Hayashi's elasticnet() implementation."""
    n, p = x.shape
    y_mean = y.mean()
    y_c = y - y_mean

    col_mean = x.mean(axis=0)
    col_std = x.std(axis=0, ddof=0)
    col_std[col_std < 1e-12] = 1.0
    x_std = (x - col_mean) / col_std

    beta = np.zeros(p)
    xx_diag = np.sum(x_std ** 2, axis=0)
    l1 = alpha * l1_ratio * n
    l2 = alpha * (1.0 - l1_ratio) * n

    for _ in range(max_iter):
        xb = x_std @ beta
        r = y_c - xb
        max_delta = 0.0
        for j in range(p):
            denom = xx_diag[j] + l2
            rho_j = x_std[:, j].dot(r) + xx_diag[j] * beta[j]
            new_b = np.sign(rho_j) * max(abs(rho_j) - l1, 0.0) / denom
            max_delta = max(max_delta, abs(new_b - beta[j]))
            beta[j] = new_b
        if max_delta < tol:
            break

    beta_orig = beta / col_std
    intercept = y_mean - beta_orig.dot(col_mean)
    return np.concatenate([[intercept], beta_orig])


params = _elasticnet_cd(X, y, alpha=0.1, l1_ratio=0.5)
coefs = {"Intercept": float(params[0])}
for name, val in zip(predictors, params[1:]):
    coefs[name] = float(val)

std_errors = {name: 0.0 for name in coefs}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

# Reference implementation in Python/statsmodels for the Wooldridge wagepan
# two-way clustered OLS case.
#
# Uses a manual Cameron-Gelbach-Miller covariance calculation so the validation
# case is not tied to package-specific finite-sample correction defaults.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "wagepan.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data

        df = data("wagepan")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/wagepan.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

formula = "lwage ~ educ + exper + expersq + union + married"
model = smf.ols(formula, data=df).fit()

row_labels = model.model.data.row_labels
model_frame = df.loc[row_labels].copy()
X = np.asarray(model.model.exog, dtype=float)
u = np.asarray(model.resid, dtype=float)


def cluster_meat(x: np.ndarray, residuals: np.ndarray, clusters: pd.Series) -> np.ndarray:
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    cluster_values = pd.Series(clusters).to_numpy()

    for cluster in pd.unique(cluster_values):
        idx = cluster_values == cluster
        xg = x[idx, :]
        ug = residuals[idx]
        score = xg.T @ ug
        meat += np.outer(score, score)

    return meat


cluster_nr = model_frame["nr"]
cluster_year = model_frame["year"]
cluster_intersection = cluster_nr.astype(str) + "::" + cluster_year.astype(str)

n, k = X.shape
g_nr = cluster_nr.nunique()
g_year = cluster_year.nunique()
g_min = min(g_nr, g_year)

xtx_inv = np.linalg.inv(X.T @ X)
meat = (
    cluster_meat(X, u, cluster_nr)
    + cluster_meat(X, u, cluster_year)
    - cluster_meat(X, u, cluster_intersection)
)

finite_sample_correction = (g_min / (g_min - 1)) * ((n - 1) / (n - k))
vcov_twoway = finite_sample_correction * xtx_inv @ meat @ xtx_inv
std_errors = np.sqrt(np.maximum(np.diag(vcov_twoway), 0.0))

coefs = {name: float(val) for name, val in model.params.items()}
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

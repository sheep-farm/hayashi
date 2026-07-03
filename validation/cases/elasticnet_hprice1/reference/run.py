# Reference implementation in Python for the Elastic Net hprice1 case.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

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
X = df[predictors].astype(float)
y = df["lprice"].astype(float)

# Match Hayashi's ElasticNet formulation: standardise X, centre y, do not
# penalise the intercept, then transform the slopes back to the original scale.
x_mean = X.mean().to_numpy()
x_std = X.std().to_numpy()
y_mean = y.mean()

X_std = ((X - x_mean) / x_std).to_numpy()
y_c = (y - y_mean).to_numpy()

model = ElasticNet(
    alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-6, fit_intercept=False
).fit(X_std, y_c)

beta_std = model.coef_
beta_orig = beta_std / x_std
intercept = float(y_mean - np.sum(beta_orig * x_mean))

coefs = {"Intercept": intercept}
for name, val in zip(predictors, beta_orig):
    coefs[name] = float(val)

result = {
    "coefficients": coefs,
    "standard_errors": {name: 0.0 for name in coefs},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

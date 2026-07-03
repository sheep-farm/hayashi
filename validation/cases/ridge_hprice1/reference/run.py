# Reference implementation in Python for the Ridge hprice1 case.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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

# Match Hayashi's ridge formulation: include the intercept in the design
# matrix and penalise it together with the slope coefficients.
X_aug = np.column_stack([np.ones(X.shape[0]), X])
model = Ridge(alpha=0.1, fit_intercept=False, max_iter=10000, tol=1e-6).fit(X_aug, y)

coefs = {"Intercept": float(model.coef_[0])}
for name, val in zip(predictors, model.coef_[1:]):
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

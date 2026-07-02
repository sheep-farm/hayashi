# Reference implementation in Python for the Lasso housing price case.

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Lasso

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
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Standardise predictors for fair comparison with glmnet/sklearn.
predictors = ["lotsize", "sqrft", "bdrms"]
X = df[predictors].astype(float)
X = (X - X.mean()) / X.std()
y = df["price"].astype(float)

# Lasso with a small penalty that does not fully shrink coefficients.
model = Lasso(alpha=100.0, max_iter=10000).fit(X, y)

coefs = {"Intercept": float(model.intercept_)}
for name, val in zip(predictors, model.coef_):
    coefs[name] = float(val)

# sklearn does not provide analytical standard errors for Lasso.
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

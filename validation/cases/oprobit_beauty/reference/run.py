# Reference implementation in Python for the ordered probit beauty case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.miscmodels.ordinal_model as om
from wooldridge import data

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "beauty.csv"

# Load Wooldridge beauty and save CSV (keep all looks categories 2-5).
df = data("beauty")
df = df[df["looks"].isin([2, 3, 4, 5])].copy()
df.to_csv(CSV_PATH, index=False)

# Ordered probit regression of looks on female, educ, exper, black.
X = df[["female", "educ", "exper", "black"]]
y = df["looks"]

model = om.OrderedModel(y, X, distr="probit").fit(method="bfgs", disp=False)

params = model.params
bse = model.bse

result = {
    "coefficients": {
        "female": float(params["female"]),
        "educ": float(params["educ"]),
        "exper": float(params["exper"]),
        "black": float(params["black"]),
    },
    "standard_errors": {
        "female": float(bse["female"]),
        "educ": float(bse["educ"]),
        "exper": float(bse["exper"]),
        "black": float(bse["black"]),
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

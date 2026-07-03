# Reference implementation in Python for the IV Card returns-to-schooling case.

import json
from pathlib import Path

import pandas as pd
from linearmodels.iv import IV2SLS

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "card.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("card")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/card.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Card returns-to-schooling IV: education endogenous, nearc4 as instrument.
vars_ = ["lwage", "educ", "exper", "expersq", "black", "south", "smsa", "nearc4"]
df = df[vars_].dropna()

endog = df[["educ"]]
exog = df[["exper", "expersq", "black", "south", "smsa"]]
exog = pd.concat([pd.Series(1, index=exog.index, name="const"), exog], axis=1)
instruments = df[["nearc4"]]
dep = df["lwage"]

model = IV2SLS(dep, exog, endog, instruments).fit(cov_type="unadjusted")

coefs = {name: float(val) for name, val in model.params.items()}
std_errors = {name: float(val) for name, val in model.std_errors.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

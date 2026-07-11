# Reference implementation in Python for Wooldridge wagepan GEE, Chapter 14.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Independence

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

vars_ = ["lwage", "union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87", "nr", "year"]
df = df[vars_].dropna()

model = smf.gee(
    "lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87",
    data=df,
    groups=df["nr"],
    cov_struct=Independence(),
    family=sm.families.Gaussian(),
).fit()

coefs = {name: float(val) for name, val in model.params.items()}
std_errors = {name: float(val) for name, val in model.bse.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
    "nobs": int(model.nobs),
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

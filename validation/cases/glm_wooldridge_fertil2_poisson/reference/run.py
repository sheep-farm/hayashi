# Reference implementation in Python for Wooldridge fertil2 Poisson GLM.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "fertil2.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("fertil2")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/fertil2.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

vars_ = ["children", "age", "electric", "educ", "urban", "tv"]
df = df[vars_].dropna()

model = smf.glm(
    "children ~ age + electric + educ + urban + tv",
    data=df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
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

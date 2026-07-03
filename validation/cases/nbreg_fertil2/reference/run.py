# Reference implementation in Python for the Negative Binomial fertil2 case.

import json
from pathlib import Path

import pandas as pd
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

# Negative binomial regression for number of children.
model = smf.negativebinomial("children ~ age + educ + electric + urban", data=df).fit(disp=0)

# Compare only the regression coefficients; the dispersion parameter (alpha)
# is not reported by Hayashi's NegBin output.
reg_vars = ["Intercept", "age", "educ", "electric", "urban"]
coefs = {name: float(model.params[name]) for name in reg_vars}
std_errors = {name: float(model.bse[name]) for name in reg_vars}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

# Reference implementation in Python for logit average marginal effects.

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "mroz.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data

        df = data("mroz")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/mroz.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

formula = "inlf ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6"
model = smf.logit(formula, data=df).fit(disp=0)
margins = model.get_margeff(at="overall", method="dydx")
frame = margins.summary_frame()

result = {
    "marginal_effects": {
        name: float(value) for name, value in frame["dy/dx"].items()
    },
    "standard_errors": {
        name: float(value) for name, value in frame["Std. Err."].items()
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

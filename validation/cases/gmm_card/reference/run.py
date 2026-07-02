# Reference implementation in Python for the GMM card case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IVGMM

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

# GMM returns-to-schooling with nearc4 as instrument for education.
model = IVGMM.from_formula(
    "lwage ~ 1 + [educ ~ nearc4] + exper + expersq + smsa + black + south",
    data=df,
).fit(cov_type="robust")

name_map = {
    "Intercept": "x0",
    "educ": "x1",
    "exper": "x2",
    "expersq": "x3",
    "smsa": "x4",
    "black": "x5",
    "south": "x6",
}
coefs = {name_map.get(name, name): float(val) for name, val in model.params.items()}
std_errors = {name_map.get(name, name): float(val) for name, val in model.std_errors.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

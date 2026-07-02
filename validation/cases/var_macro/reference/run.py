# Reference implementation in Python for the VAR macro case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "macrodata.csv"

if not CSV_PATH.exists():
    macro = sm.datasets.macrodata.load_pandas().data
    macro = macro[["year", "quarter", "realgdp", "realcons"]]
    macro = macro.rename(columns={"realgdp": "gdp", "realcons": "cons"})
    macro.to_csv(CSV_PATH, index=False)
else:
    macro = pd.read_csv(CSV_PATH)

# VAR(2) on GDP and consumption.
model = VAR(macro[["gdp", "cons"]].astype(float).dropna()).fit(maxlags=2)

coefs = {}
std_errors = {}
for eq_name in model.params.columns:
    for idx, row_name in enumerate(model.params.index):
        key = f"{eq_name}_{row_name}"
        coefs[key] = float(model.params.loc[row_name, eq_name])
        std_errors[key] = float(model.stderr.loc[row_name, eq_name])

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

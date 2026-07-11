# Reference implementation in Python for robust linear model on Wooldridge wage1.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from wooldridge import data

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "wage1.csv"

df = data("wage1")
df.to_csv(CSV_PATH, index=False)

# Huber robust regression via RLM in statsmodels
model = smf.rlm("lwage ~ educ + exper + tenure", data=df, M=sm.robust.norms.HuberT()).fit()

params = model.params
bse = model.bse

result = {
    "coefficients": {name: float(params[name]) for name in params.index},
    "standard_errors": {name: float(bse[name]) for name in bse.index},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

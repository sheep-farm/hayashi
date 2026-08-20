# Reference implementation in Python/statsmodels for Wooldridge bwght,
# Chapter 5, Example 5.2.

import json
import os
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "bwght.csv"

# Load the same CSV that Hayashi will read, or generate it if absent.
if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("bwght")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/bwght.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Estimate the model from Chapter 5, Example 5.2.
model = smf.ols("lbwght ~ 1 + cigs + lfaminc", data=df).fit()

coefs = {name: float(val) for name, val in model.params.items()}
std_errors = {name: float(val) for name, val in model.bse.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
    "r_squared": float(model.rsquared),
    "nobs": int(model.nobs),
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

# Write JSON for the orchestrator to compare.
with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
print(json.dumps(result))

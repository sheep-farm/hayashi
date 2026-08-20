# Reference implementation in Python/statsmodels for Wooldridge 401k,
# Chapter 3, Example 3.3.

import json
import os
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "401k.csv"

# Load the same CSV that Hayashi will read, or generate it if absent.
if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("401k")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/401k.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Estimate the 401(k) participation equation from Example 3.3.
model = smf.ols("prate ~ 1 + mrate + age", data=df).fit()

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

# Reference implementation in Python/statsmodels for Wooldridge phillips,
# Chapter 11, Example 11.5.

import json
import os
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "phillips.csv"

# Load the same CSV that Hayashi will read, or generate it if absent.
if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("phillips")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/phillips.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Drop rows with missing values for any model variable to match the textbook sample.
df = df.dropna(subset=["cinf", "unem"])

# Estimate the model from Chapter 11, Example 11.5.
model = smf.ols("cinf ~ 1 + unem", data=df).fit()

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

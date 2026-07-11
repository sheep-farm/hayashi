# Reference implementation in Python for Wooldridge wagepan panel FE, Chapter 14, Example 14.4.

import json
from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS

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

# Keep only observations with complete model variables.
vars_ = ["lwage", "union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87", "nr", "year"]
df = df[vars_].dropna()

# Set the panel index.
panel_df = df.set_index(["nr", "year"])

# Fixed-effects wage equation.
model = PanelOLS.from_formula("lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87 + EntityEffects", data=panel_df).fit()

coefs = {name: float(val) for name, val in model.params.items()}
std_errors = {name: float(val) for name, val in model.std_errors.items()}

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

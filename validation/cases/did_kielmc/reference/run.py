# Reference implementation in Python for the DiD Kiel-McClain case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "kielmc.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("kielmc")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/kielmc.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Difference-in-differences via interaction.
model = smf.ols("lprice ~ nearinc * y81", data=df).fit()

name_map = {
    "Intercept": "const",
    "nearinc": "treated",
    "y81": "post",
    "nearinc:y81": "treated:post",
}
coefs = {name_map.get(name, name): float(val) for name, val in model.params.items()}
std_errors = {name_map.get(name, name): float(val) for name, val in model.bse.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

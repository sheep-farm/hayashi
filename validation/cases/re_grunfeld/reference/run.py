# Reference implementation in Python for the random-effects Grunfeld case.

import json
from pathlib import Path

import pandas as pd
from linearmodels.panel import RandomEffects

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "grunfeld.csv"

if not CSV_PATH.exists():
    import statsmodels.datasets.grunfeld as grunfeld_module
    df = grunfeld_module.load_pandas().data
    df = df.rename(columns={"invest": "inv"})
    df["firm"] = pd.factorize(df["firm"])[0] + 1
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

panel_df = df.set_index(["firm", "year"])

model = RandomEffects.from_formula("inv ~ 1 + value + capital", data=panel_df).fit()

coefs = {name: float(val) for name, val in model.params.items()}
std_errors = {name: float(val) for name, val in model.std_errors.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

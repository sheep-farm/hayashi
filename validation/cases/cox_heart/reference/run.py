# Reference implementation in Python for the Cox survival heart case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "heart.csv"

if not CSV_PATH.exists():
    heart = sm.datasets.heart.load_pandas().data
    heart = heart.rename(columns={"censors": "censored", "survival": "time"})
    heart.to_csv(CSV_PATH, index=False)
else:
    heart = pd.read_csv(CSV_PATH)

# Cox proportional hazards: survival time after heart transplant.
model = sm.PHReg(
    endog=heart["time"].astype(float).to_numpy(),
    exog=heart["age"].astype(float).to_numpy().reshape(-1, 1),
    status=heart["censored"].astype(int).to_numpy(),
).fit()

coefs = {"age": float(model.params[0])}
std_errors = {"age": float(model.bse[0])}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

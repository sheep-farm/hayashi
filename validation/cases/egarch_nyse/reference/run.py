# Reference implementation in Python for the EGARCH NYSE returns case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from arch import arch_model

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "nyse.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("nyse")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/nyse.csv"
        df = pd.read_csv(url)
    df = df[["price", "return"]].dropna()
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# EGARCH(1,1) on NYSE returns.
returns = df["return"].astype(float).dropna()
model = arch_model(returns, vol="EGarch", p=1, q=1, rescale=False).fit(disp="off")

coefs = {name: float(val) for name, val in model.params.items()}
std_errors = {name: float(val) for name, val in model.std_err.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

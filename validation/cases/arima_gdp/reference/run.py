# Reference implementation in Python for the ARIMA GDP case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "macrodata.csv"

if not CSV_PATH.exists():
    macro = sm.datasets.macrodata.load_pandas().data
    macro = macro[["year", "quarter", "realgdp"]]
    macro = macro.rename(columns={"realgdp": "gdp"})
    macro.to_csv(CSV_PATH, index=False)
else:
    macro = pd.read_csv(CSV_PATH)

macro["lgdp"] = macro["gdp"].astype(float).apply(lambda x: float(x))

# ARIMA(1,1,1) on log GDP.
model = ARIMA(macro["lgdp"], order=(1, 1, 1)).fit()

coefs = {name: float(val) for name, val in model.params.items()}
std_errors = {name: float(val) for name, val in model.bse.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

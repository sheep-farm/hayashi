# Reference implementation in Python for the autoregressive GDP case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

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

# AutoReg(1) on GDP with constant and trend.
model = AutoReg(macro["gdp"].astype(float), lags=1, trend="ct").fit()

coefs = {name.replace("gdp.L1", "y.L1"): float(val) for name, val in model.params.items()}
std_errors = {name.replace("gdp.L1", "y.L1"): float(val) for name, val in model.bse.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

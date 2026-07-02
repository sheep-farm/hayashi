# Reference implementation in Python for the ARDL GDP case.

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
    macro = macro[["year", "quarter", "realgdp", "realcons"]]
    macro = macro.rename(columns={"realgdp": "gdp", "realcons": "cons"})
    macro.to_csv(CSV_PATH, index=False)
else:
    macro = pd.read_csv(CSV_PATH)

# ARDL(1,1): y_t on y_{t-1}, x_t and x_{t-1}.
macro = macro.dropna(subset=["gdp", "cons"])
y = macro["gdp"].astype(float)
x = macro["cons"].astype(float)
x_lag = x.shift(1).rename("cons_lag")
exog = pd.concat([x, x_lag], axis=1).dropna()
y = y.loc[exog.index]

model = AutoReg(y, lags=1, exog=exog, trend="c").fit()

name_map = {
    "const": "const",
    "gdp.L1": "y.L1",
    "cons": "x1",
    "cons_lag": "x1.L1",
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

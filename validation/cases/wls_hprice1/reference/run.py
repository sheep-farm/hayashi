# Reference implementation in Python/statsmodels for the WLS hprice1 case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "hprice1.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("hprice1")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/hprice1.csv"
        df = pd.read_csv(url)
    # Write the raw dataset; Hayashi will generate the weight column itself.
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Weight assumption: error variance proportional to lotsize.
df["w"] = 1.0 / df["lotsize"]

model = smf.wls("price ~ lotsize + sqrft + bdrms", data=df, weights=df["w"]).fit()

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

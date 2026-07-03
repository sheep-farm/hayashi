# Reference implementation in Python for the GLSAR housing price case.

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import GLSAR

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
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# GLSAR(1) on housing price equation.
# Use iterative_fit so statsmodels actually estimates the AR(1) rho,
# matching Hayashi's Cochrane-Orcutt procedure.
model = GLSAR.from_formula("price ~ lotsize + sqrft + bdrms", data=df, rho=1)
res = model.iterative_fit(maxiter=10)

coefs = {name: float(val) for name, val in res.params.items()}
std_errors = {name: float(val) for name, val in res.bse.items()}

result = {
    "coefficients": coefs,
    "standard_errors": std_errors,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

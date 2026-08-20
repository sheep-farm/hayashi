"""Python reference implementation for ZIP on Wooldridge affairs."""

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson


CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
REF_DIR = CASE_DIR / "reference"
CSV_PATH = DATA_DIR / "affairs.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
else:
    try:
        from wooldridge import data

        df = data("affairs")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/affairs.csv"
        df = pd.read_csv(url)
        if "rownames" in df.columns:
            df = df.drop(columns=["rownames"])
    df.to_csv(CSV_PATH, index=False)

variables = ["naffairs", "age", "yrsmarr", "kids", "educ", "relig", "ratemarr"]
df = df[variables].dropna()

regressors = ["age", "yrsmarr", "kids", "educ", "relig", "ratemarr"]
x_count = sm.add_constant(df[regressors], has_constant="add")
x_inflate = sm.add_constant(df[regressors], has_constant="add")

model = ZeroInflatedPoisson(
    df["naffairs"],
    x_count,
    exog_infl=x_inflate,
    inflation="logit",
)
fit = model.fit(method="bfgs", maxiter=1000, disp=0)

if not fit.mle_retvals.get("converged", False):
    raise RuntimeError("statsmodels ZeroInflatedPoisson did not converge")

names = ["const", *regressors]
coefficients = {}
standard_errors = {}

for name in names:
    coefficients[f"count_{name}"] = float(fit.params[name])
    standard_errors[f"count_{name}"] = float(fit.bse[name])

for name in names:
    inflate_name = f"inflate_{name}"
    coefficients[inflate_name] = float(fit.params[inflate_name])
    standard_errors[inflate_name] = float(fit.bse[inflate_name])

result = {
    "coefficients": coefficients,
    "standard_errors": standard_errors,
}

out = json.dumps(result)
(REF_DIR / "expected.json").write_text(out, encoding="utf-8")
print(out)

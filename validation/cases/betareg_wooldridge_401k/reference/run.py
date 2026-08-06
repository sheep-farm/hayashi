"""Python reference implementation for beta regression on Wooldridge 401k."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel


CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
REF_DIR = CASE_DIR / "reference"
CSV_PATH = DATA_DIR / "401k.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
else:
    try:
        from wooldridge import data

        df = data("k401k")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/k401k.csv"
        df = pd.read_csv(url)
        if "rownames" in df.columns:
            df = df.drop(columns=["rownames"])

    df["prate"] = df["prate"] / 100.0
    df["prate"] = df["prate"].clip(1e-4, 1.0 - 1e-4)
    df.to_csv(CSV_PATH, index=False)

variables = ["prate", "mrate", "age", "sole"]
df = df[variables].dropna()

if not ((df["prate"] > 0.0) & (df["prate"] < 1.0)).all():
    raise RuntimeError("beta-regression response must be strictly inside (0, 1)")

regressors = ["mrate", "age", "sole"]
names = ["const", *regressors]
x_mean = sm.add_constant(df[regressors], has_constant="add")

model = BetaModel(df["prate"], x_mean)
fit = model.fit(method="bfgs", maxiter=1000, disp=0)

if not fit.mle_retvals.get("converged", False):
    raise RuntimeError("statsmodels BetaModel did not converge")

# The validation contract compares the mean equation only; the precision
# intercept is intentionally left out to match the existing R reference.
coefficients = {name: float(fit.params[name]) for name in names}
standard_errors = {name: float(fit.bse[name]) for name in names}

values = [*coefficients.values(), *standard_errors.values(), float(fit.llf)]
if not all(np.isfinite(values)):
    raise RuntimeError("statsmodels BetaModel produced non-finite output")

result = {
    "coefficients": coefficients,
    "standard_errors": standard_errors,
}

out = json.dumps(result)
(REF_DIR / "expected.json").write_text(out, encoding="utf-8")
print(out)

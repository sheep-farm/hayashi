# Reference implementation in Python for the ETS GDP case.
#
# Uses simple exponential smoothing (SES, ETS(A,N,N)) to match the Hayashi
# `ses(df, gdp)` call.  Only alpha is reported because the Hayashi text output
# exposes only the smoothing parameter.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

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

y = macro["gdp"].astype(float).values

# SES with optimised smoothing parameter.
model = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
alpha = float(model.params["smoothing_level"])

# Clip tiny negative values that can appear at the boundary.
alpha = max(0.0, min(1.0, alpha))

result = {
    "coefficients": {"alpha": alpha},
    "standard_errors": {"alpha": 0.0},
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

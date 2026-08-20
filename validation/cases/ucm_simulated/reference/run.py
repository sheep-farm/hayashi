import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace import structural

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "data.csv"

df = pd.read_csv(DATA_PATH)
y = df["y"].values

mod = structural.UnobservedComponents(
    y, level="local linear trend", seasonal=12, stochastic_seasonal=False
)
res = mod.fit(disp=False)

out = {
    "coefficients": {
        "sigma2_irregular": float(res.params[0]),
        "level_first": float(res.level["smoothed"][0]),
        "level_last": float(res.level["smoothed"][-1]),
    },
    "standard_errors": {
        "sigma2_irregular": 0.0,
        "level_first": 0.0,
        "level_last": 0.0,
    },
}

print(json.dumps(out, indent=2))

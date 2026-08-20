import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "data.csv"

df = pd.read_csv(DATA_PATH)
y = df["y"].values

n = len(y)
period = 12
first = 6
last = n - period // 2 - 1

result = seasonal_decompose(y, model="additive", period=period, extrapolate_trend=0)

out = {
    "coefficients": {
        "trend_first": float(result.trend[first]),
        "trend_last": float(result.trend[last]),
        "seasonal_first": float(result.seasonal[first]),
        "seasonal_last": float(result.seasonal[last]),
        "resid_first": float(result.resid[first]),
        "resid_last": float(result.resid[last]),
    },
    "standard_errors": {
        "trend_first": float("nan"),
        "trend_last": float("nan"),
        "seasonal_first": float("nan"),
        "seasonal_last": float("nan"),
        "resid_first": float("nan"),
        "resid_last": float("nan"),
    },
}

print(json.dumps(out, indent=2))

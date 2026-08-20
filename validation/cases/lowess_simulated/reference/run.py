import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

CASE_DIR = Path("validation/cases/lowess_simulated")
DATA_DIR = CASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "data.csv")
fit = lowess(df["y"].values, df["x"].values, frac=0.4, it=3, return_sorted=False)

result = {
    "coefficients": {
        "mean_yhat": float(fit.mean()),
        "yhat_first": float(fit[0]),
        "yhat_mid": float(fit[99]),
        "yhat_last": float(fit[199]),
    },
    "standard_errors": {
        "mean_yhat": 0.0,
        "yhat_first": 0.0,
        "yhat_mid": 0.0,
        "yhat_last": 0.0,
    },
}

print(json.dumps(result))

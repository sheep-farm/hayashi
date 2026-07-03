# Reference implementation in Python for the ARIMA(1,1,0) book case.
# Uses statsmodels.tsa.statespace.SARIMAX with no trend, matching the R
# reference and the Hayashi ARIMA output (which still reports an intercept,
# but the intercept is ignored in the validation comparison).

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "arima.csv"

df = pd.read_csv(DATA_PATH)

m = sm.tsa.statespace.SARIMAX(df["rw"], order=(1, 1, 0), trend="n").fit(disp=False)

result = {
    "coefficients": {
        "ar.L1": float(m.params["ar.L1"]),
    },
    "standard_errors": {
        "ar.L1": float(m.bse["ar.L1"]),
    },
}

print(json.dumps(result))

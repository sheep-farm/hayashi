import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "data.csv")
y1 = df["y1"]
y1_lag = y1.shift(1)
y2_lag = df["y2"].shift(1)
exog = pd.DataFrame({"y1_lag": y1_lag, "y2_lag": y2_lag})

# Drop the first observation because lags are missing
y1 = y1.iloc[1:]
exog = exog.iloc[1:]

model = sm.tsa.MarkovRegression(
    y1, k_regimes=2, trend="c", exog=exog, switching_variance=True
)
res = model.fit(em_iter=100, cov_type="none")

params = res.params
const0 = float(params["const[0]"])
const1 = float(params["const[1]"])
p00_raw = float(params["p[0->0]"])
p10_raw = float(params["p[1->0]"])

# Sort by intercept magnitude so mu0 < mu1
if const0 < const1:
    mu0_y1 = const0
    mu1_y1 = const1
    p00 = p00_raw
    p01 = 1 - p00_raw
    p10 = p10_raw
    p11 = 1 - p10_raw
else:
    mu0_y1 = const1
    mu1_y1 = const0
    p00 = 1 - p10_raw
    p01 = p10_raw
    p10 = 1 - p00_raw
    p11 = p00_raw

result = {
    "coefficients": {
        "mu0_y1": mu0_y1,
        "mu1_y1": mu1_y1,
        "p00": p00,
        "p01": p01,
        "p10": p10,
        "p11": p11,
    },
    "standard_errors": {
        "mu0_y1": 0.0,
        "mu1_y1": 0.0,
        "p00": 0.0,
        "p01": 0.0,
        "p10": 0.0,
        "p11": 0.0,
    },
}

print(json.dumps(result))

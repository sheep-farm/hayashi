import json
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm


df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

model = sm.tsa.VARMAX(df[["y1", "y2"]], order=(1, 1), trend="n")
fit = model.fit(disp=False)

p = fit.params

# Hayashi vectorises the VARMA parameters in column-major order.
# AR matrix Phi (rows = equations y1, y2; cols = variables y1, y2):
#   p11 = L1.y1.y1   p12 = L1.y2.y1
#   p21 = L1.y1.y2   p22 = L1.y2.y2
# Column-major: [p11, p21, p12, p22]
# The first two entries (ar0, ar1) are intercepts, which are zero with trend='n'.
# MA matrix Theta (rows = equations; cols = shocks e(y1), e(y2)):
#   t11 = L1.e(y1).y1  t12 = L1.e(y2).y1
#   t21 = L1.e(y1).y2  t22 = L1.e(y2).y2
# Column-major: [t11, t21, t12, t22]

coef = {
    "ar0": 0.0,
    "ar1": 0.0,
    "ar2": float(p["L1.y1.y1"]),
    "ar3": float(p["L1.y1.y2"]),
    "ar4": float(p["L1.y2.y1"]),
    "ar5": float(p["L1.y2.y2"]),
    "ma0": float(p["L1.e(y1).y1"]),
    "ma1": float(p["L1.e(y1).y2"]),
    "ma2": float(p["L1.e(y2).y1"]),
    "ma3": float(p["L1.e(y2).y2"]),
}

result = {"coefficients": coef}
print(json.dumps(result, indent=2))

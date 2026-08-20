import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.vector_ar.vecm import coint_johansen

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
Y = df[["y1", "y2"]].values
shift = df["shift"].values.astype(float)

# Adjust for the structural break by removing the break dummy from the levels,
# then run the standard Johansen test.  This is an approximation because
# statsmodels.tsa.vecm.coint_johansen does not accept exogenous dummy variables.
X = np.column_stack([np.ones(len(df)), shift])
beta = np.linalg.lstsq(X, Y, rcond=None)[0]
resid = Y - X @ beta

res = coint_johansen(resid, det_order=0, k_ar_diff=1)

trace = res.lr1
rank = int(trace[0] > res.cvt[0, 1])

result = {
    "coefficients": {
        "rank": float(rank),
        "trace_0": float(trace[0]),
        "trace_1": float(trace[1]),
    },
    "standard_errors": {"rank": 0.0, "trace_0": 0.0, "trace_1": 0.0},
}

print(json.dumps(result, indent=2))

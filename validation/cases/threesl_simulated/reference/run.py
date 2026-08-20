import json
import numpy as np
import pandas as pd
from pathlib import Path
from linearmodels.system import IV3SLS

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

# linearmodels IV3SLS does not add a constant by default, so we add one
# explicitly to match Hayashi's default 3SLS specification.
df["const"] = 1.0

# Equation y1: y1 = a1*x1 + a2*y2 + u1, instrument y2 with x2
# Equation y2: y2 = b1*x2 + b2*y1 + u2, instrument y1 with x1
eq1 = {
    "dependent": df[["y1"]],
    "exog": df[["const", "x1"]],
    "endog": df[["y2"]],
    "instruments": df[["x2"]],
}
eq2 = {
    "dependent": df[["y2"]],
    "exog": df[["const", "x2"]],
    "endog": df[["y1"]],
    "instruments": df[["x1"]],
}

mod = IV3SLS({"y1": eq1, "y2": eq2})
res = mod.fit(cov_type="unadjusted")

params = res.params
stderr = res.std_errors

result = {
    "coefficients": {
        "y1:const": float(params["y1_const"]),
        "y1:x1": float(params["y1_x1"]),
        "y1:y2": float(params["y1_y2"]),
        "y2:const": float(params["y2_const"]),
        "y2:x2": float(params["y2_x2"]),
        "y2:y1": float(params["y2_y1"]),
    },
    "standard_errors": {
        "y1:const": float(stderr["y1_const"]),
        "y1:x1": float(stderr["y1_x1"]),
        "y1:y2": float(stderr["y1_y2"]),
        "y2:const": float(stderr["y2_const"]),
        "y2:x2": float(stderr["y2_x2"]),
        "y2:y1": float(stderr["y2_y1"]),
    },
}

print(json.dumps(result, indent=2))

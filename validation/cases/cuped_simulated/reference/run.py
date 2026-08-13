import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
t = df["treated"].values
pre = df["pre_y"].values
y = df["y"].values
# Match R's cov()/var() sample-moment convention (both use n - 1).
theta = np.cov(y, pre)[0, 1] / np.var(pre, ddof=1)
y_adj = y - theta * (pre - pre.mean())
ate = float(y_adj[t == 1].mean() - y_adj[t == 0].mean())
unadj = float(y[t == 1].mean() - y[t == 0].mean())
vr = 1.0 - np.var(y_adj) / np.var(y)
result = {
  "coefficients": {
    "treatment_effect": ate,
    "unadjusted_effect": unadj,
    "theta": float(theta),
    "variance_reduction": float(vr)
  },
  "standard_errors": {"treatment_effect": float("nan"), "unadjusted_effect": float("nan"), "theta": float("nan"), "variance_reduction": float("nan")}
}
print(json.dumps(result))

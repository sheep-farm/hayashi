import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

# Within transformation (entity-demeaning)
df["y_dem"] = df.groupby("id")["y"].transform(lambda s: s - s.mean())
df["x_dem"] = df.groupby("id")["x"].transform(lambda s: s - s.mean())

# Quantile regression without intercept (fixed effects absorbed)
model = smf.quantreg("y_dem ~ x_dem - 1", df)
res = model.fit(q=0.75)

result = {
    "coefficients": {
        "x": float(res.params["x_dem"]),
    },
    "standard_errors": {
        "x": float(res.bse["x_dem"]),
    },
}

print(json.dumps(result, indent=2))

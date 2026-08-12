import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

case_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(case_dir / "data" / "data.csv")

for t in (-2, 0, 1, 2):
    df[f"t_{t}"] = (df["event_time"] == t).astype(int)

X = sm.add_constant(df[["t_-2", "t_0", "t_1", "t_2"]])
y = df["y"].values

res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["unit"]})

names = ["t=-2", "t=0", "t=1", "t=2"]
params = res.params[1:].values
bse = res.bse[1:].values

coefficients = {name: float(params[i]) for i, name in enumerate(names)}
standard_errors = {name: float(bse[i]) for i, name in enumerate(names)}

result = {"coefficients": coefficients, "standard_errors": standard_errors}
print(json.dumps(result, indent=2))

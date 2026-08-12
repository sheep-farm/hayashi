import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

window = 20
last = df.iloc[-window:]
X = sm.add_constant(last["x"].values)
y = last["y"].values

model = sm.OLS(y, X)
res = model.fit()

result = {
    "coefficients": {
        "const": float(res.params[0]),
        "x": float(res.params[1]),
    },
    "standard_errors": {
        "const": 0.0,
        "x": 0.0,
    },
}

print(json.dumps(result, indent=2))

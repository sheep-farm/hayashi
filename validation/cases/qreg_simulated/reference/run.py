import json
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

model = smf.quantreg("y ~ x", df)
res = model.fit(q=0.75)

result = {
    "coefficients": {
        "const": float(res.params["Intercept"]),
        "x": float(res.params["x"]),
    },
    "standard_errors": {
        "const": float(res.bse["Intercept"]),
        "x": float(res.bse["x"]),
    },
}

print(json.dumps(result, indent=2))

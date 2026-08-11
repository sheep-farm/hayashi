import json
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

model = smf.quantreg("y ~ x", df)
res = model.fit(q=0.75)

result = {
    "const": float(res.params["Intercept"]),
    "x": float(res.params["x"]),
}

print(json.dumps(result, indent=2))

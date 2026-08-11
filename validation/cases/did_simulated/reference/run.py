import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

model = smf.ols("y ~ treat + post + treat:post", data=df)
res = model.fit(cov_type="HC0")

result = {
    "coefficients": {
        "treated:post": float(res.params["treat:post"]),
    },
}

print(json.dumps(result, indent=2))

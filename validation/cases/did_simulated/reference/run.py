import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

model = smf.ols("y ~ treat + post + treat:post", data=df)
res = model.fit(cov_type="HC0")

coef = res.params
se = res.bse

result = {
    "coefficients": {
        "const": float(coef["Intercept"]),
        "treated": float(coef["treat"]),
        "post": float(coef["post"]),
        "treated:post": float(coef["treat:post"]),
    },
    "standard_errors": {
        "const": float(se["Intercept"]),
        "treated": float(se["treat"]),
        "post": float(se["post"]),
        "treated:post": float(se["treat:post"]),
    },
}

print(json.dumps(result, indent=2))

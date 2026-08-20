import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

model = smf.gee(
    "y ~ x",
    groups="id",
    data=df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable(),
)
res = model.fit()

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

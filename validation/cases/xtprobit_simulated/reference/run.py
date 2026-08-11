import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

model = smf.gee(
    "y ~ x",
    groups="id",
    data=df,
    family=sm.families.Binomial(link=sm.families.links.Probit()),
    cov_struct=sm.cov_struct.Exchangeable(),
)
res = model.fit()

result = {
    "coefficients": {
        "const": float(res.params["Intercept"]),
        "x": float(res.params["x"]),
    },
}

print(json.dumps(result, indent=2))

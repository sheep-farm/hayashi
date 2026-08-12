import json
import pandas as pd
import statsmodels.api as sm

CASE_DIR = "validation/cases/bayes_lm_simulated"
DATA_DIR = f"{CASE_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/data.csv")
X = sm.add_constant(df[["x1", "x2"]])
mod = sm.OLS(df["y"], X).fit()

result = {
    "coefficients": {
        "b1": float(mod.params["x1"]),
        "b2": float(mod.params["x2"]),
    },
    "standard_errors": {
        "b1": float(mod.bse["x1"]),
        "b2": float(mod.bse["x2"]),
    },
}

print(json.dumps(result))

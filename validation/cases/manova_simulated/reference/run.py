import json
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

CASE_DIR = "validation/cases/manova_simulated"
DATA_DIR = f"{CASE_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/data.csv")
df["group"] = df["group"].astype("category")

mv = MANOVA.from_formula("y1 + y2 ~ group", data=df)
tbl = mv.mv_test().results["group"]["stat"]

result = {
    "coefficients": {
        "pillai": float(tbl.loc["Pillai's trace", "Value"]),
        "wilks": float(tbl.loc["Wilks' lambda", "Value"]),
        "lh": float(tbl.loc["Hotelling-Lawley trace", "Value"]),
        "roy": float(tbl.loc["Roy's greatest root", "Value"]),
    },
    "standard_errors": {"pillai": 0.0, "wilks": 0.0, "lh": 0.0, "roy": 0.0},
}

print(json.dumps(result))

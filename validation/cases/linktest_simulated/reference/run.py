import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].to_numpy()
model = smf.logit("y ~ x", data=df).fit(disp=0)
X = sm.add_constant(df["x"].to_numpy(), has_constant="add")
yhat = X @ model.params.to_numpy()
X2 = np.column_stack([yhat, yhat ** 2])
logit = sm.Logit(y, X2).fit(disp=0)

result = {
    "coefficients": {"yhat": float(logit.params[0]), "yhat2": float(logit.params[1])},
    "standard_errors": {"yhat": float(logit.bse[0]), "yhat2": float(logit.bse[1])},
}
print(json.dumps(result))

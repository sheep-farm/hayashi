import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.regression.quantile_regression import QuantReg

case_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(case_dir / "data" / "data.csv")

Y = df.iloc[1:].reset_index(drop=True)
Ylag = df.iloc[:-1].reset_index(drop=True)

coefs = {}
for v in ("y1", "y2"):
    X = np.column_stack([
        np.ones(len(Y)),
        Ylag["y1"].values,
        Ylag["y2"].values,
    ])
    y = Y[v].values
    model = QuantReg(y, X)
    res = model.fit(q=0.5)
    names = ["const", "L1.y1", "L1.y2"]
    for p, b in zip(names, res.params):
        coefs[f"{v}_{p}"] = float(b)

result = {"coefficients": coefs}
print(json.dumps(result, indent=2))

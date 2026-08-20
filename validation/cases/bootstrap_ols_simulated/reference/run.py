import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

np.random.seed(42)
n = len(df)
B = 2000
n_boot = n

coefs = np.zeros((B, 2))

X = sm.add_constant(df["x"].values)
y = df["y"].values

for b in range(B):
    idx = np.random.choice(n, size=n_boot, replace=True)
    Xb = X[idx]
    yb = y[idx]
    model = sm.OLS(yb, Xb)
    res = model.fit()
    coefs[b, 0] = res.params[0]
    coefs[b, 1] = res.params[1]

means = coefs.mean(axis=0)
sds = coefs.std(axis=0, ddof=1)

result = {
    "coefficients": {
        "const": float(means[0]),
        "x": float(means[1]),
    },
    "standard_errors": {
        "const": float(sds[0]),
        "x": float(sds[1]),
    },
}

print(json.dumps(result, indent=2))

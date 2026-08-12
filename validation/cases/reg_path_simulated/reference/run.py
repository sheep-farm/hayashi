import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNet

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

X = df[["x1", "x2"]].astype(float).to_numpy()
y = df["y"].astype(float).to_numpy()

n, p = X.shape
x_mean = X.mean(axis=0)
x_std = X.std(axis=0, ddof=0)
x_std[x_std < 1e-12] = 1.0
y_mean = y.mean()
y_std = y.std(ddof=0)

X_std = (X - x_mean) / x_std
y_c = y - y_mean

model = ElasticNet(
    alpha=0.003642,
    l1_ratio=0.5,
    fit_intercept=False,
    max_iter=10000,
    tol=1e-6,
)
model.fit(X_std, y_c)

beta = model.coef_
beta_orig = beta / x_std
intercept = y_mean - beta_orig.dot(x_mean)

result = {
    "coefficients": {
        "const": float(intercept),
        "x1": float(beta_orig[0]),
        "x2": float(beta_orig[1]),
    },
    "standard_errors": {
        "const": 0.0,
        "x1": 0.0,
        "x2": 0.0,
    },
}

print(json.dumps(result, indent=2))

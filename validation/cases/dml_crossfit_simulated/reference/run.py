import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

CASE_DIR = "validation/cases/dml_crossfit_simulated"
DATA_DIR = f"{CASE_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/data.csv")
df["d"] = df["dvar"] - 0.5
X = df[["x1", "x2"]].values
y = df["y"].values
d = df["d"].values

n = len(df)
y_tilde = np.zeros(n)
d_tilde = np.zeros(n)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    gy = LinearRegression().fit(X[train_idx], y[train_idx])
    gd = LinearRegression().fit(X[train_idx], d[train_idx])
    y_tilde[test_idx] = y[test_idx] - gy.predict(X[test_idx])
    d_tilde[test_idx] = d[test_idx] - gd.predict(X[test_idx])

# OLS of y_tilde on d_tilde without intercept
mod = LinearRegression(fit_intercept=False).fit(d_tilde.reshape(-1, 1), y_tilde)
theta = float(mod.coef_[0])
# residual SE
y_pred = d_tilde * theta
resid = y_tilde - y_pred
rss = np.sum(resid**2)
se = float(np.sqrt(rss / (n - 1) / np.sum(d_tilde**2)))

result = {
    "coefficients": {"theta": theta},
    "standard_errors": {"theta": se},
}

print(json.dumps(result))

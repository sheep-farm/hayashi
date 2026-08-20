import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

CASE_DIR = "validation/cases/conformal_simulated"
DATA_DIR = f"{CASE_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/data.csv")
n = len(df)
calib_n = int(0.3 * n)

train = df.iloc[: n - calib_n].copy()
calib = df.iloc[n - calib_n :].copy()

X_train = sm.add_constant(train[["x1", "x2"]])
model = sm.OLS(train["y"], X_train).fit()

X_calib = sm.add_constant(calib[["x1", "x2"]])
mu_calib = model.predict(X_calib)
scores = np.abs(calib["y"].values - mu_calib.values)
q = np.quantile(scores, 0.9)

X_test = sm.add_constant(df[["x1", "x2"]])
mu_test = model.predict(X_test)
lower = mu_test - q
upper = mu_test + q
coverage = np.mean((df["y"] >= lower) & (df["y"] <= upper))

result = {
    "coefficients": {
        "empirical_coverage": float(coverage),
        "conformal_quantile": float(q),
    },
    "standard_errors": {"empirical_coverage": 0.0, "conformal_quantile": 0.0},
}

print(json.dumps(result))

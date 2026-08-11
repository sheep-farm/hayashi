import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")

# Reshape to matrix (units x periods)
units = df["unit"].unique()
periods = df["period"].unique()
T = len(periods)
n_units = len(units)

Y = np.zeros((n_units, T))
for _, row in df.iterrows():
    i = int(row["unit"])
    t = int(row["period"])
    Y[i, t] = row["y"]

treated = (df.groupby("unit")["treated"].max() > 0).values
treat_period = int(df[df["treated"] == 1]["period"].min())

treated_idx = np.where(treated)[0]
control_idx = np.where(~treated)[0]

# Average treated unit(s)
y_treated = Y[treated_idx].mean(axis=0)

# Control matrix
Y_pre = Y[control_idx, :treat_period]
y_treated_pre = y_treated[:treat_period]

# Find weights minimizing pre-treatment MSE
w0 = np.ones(len(control_idx)) / len(control_idx)


def mse(w):
    y_syn = w.dot(Y_pre)
    return np.mean((y_treated_pre - y_syn) ** 2)


cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
bounds = [(0, 1)] * len(control_idx)
res = minimize(mse, w0, method="SLSQP", bounds=bounds, constraints=cons)
w = res.x

# ATT = average post-treatment gap
y_syn_post = w.dot(Y[control_idx, treat_period:])
y_treated_post = y_treated[treat_period:]
att = (y_treated_post - y_syn_post).mean()

result = {
    "coefficients": {
        "ATT": float(att),
    },
    "standard_errors": {
        "ATT": float("nan"),
    },
}

print(json.dumps(result, indent=2))

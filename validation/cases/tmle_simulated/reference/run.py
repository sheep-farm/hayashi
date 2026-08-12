import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "data.csv")
Y = df["y"].values
A = df["t"].values
W = df[["x1", "x2"]].values

# propensity score
g_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
g_model.fit(W, A)
g1 = g_model.predict_proba(W)[:, 1]

# initial outcome regression Q(A, W)
X = np.column_stack([A, W])
q_model = LinearRegression().fit(X, Y)
Q1 = q_model.predict(np.column_stack([np.ones_like(A), W]))
Q0 = q_model.predict(np.column_stack([np.zeros_like(A), W]))
Q = A * Q1 + (1 - A) * Q0

# clever covariate and fluctuation parameter
H1 = A / g1
H0 = (1 - A) / (1 - g1)
H = H1 - H0
epsilon = np.sum(H * (Y - Q)) / np.sum(H * H)

# targeted outcome
Q1s = Q1 + epsilon * (1 / g1)
Q0s = Q0 + epsilon * (-1 / (1 - g1))
ate = float(np.mean(Q1s - Q0s))

# efficient influence function standard error
eif = H * (Y - (A * Q1s + (1 - A) * Q0s)) + Q1s - Q0s - ate
se = float(np.sqrt(np.var(eif, ddof=1) / len(Y)))

result = {
    "coefficients": {"ate": ate, "se": se},
    "standard_errors": {"ate": 0.0, "se": 0.0},
}

print(json.dumps(result))

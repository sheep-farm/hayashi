import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].values
X = df[["x1", "x2"]].values

mlp = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="logistic",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=200,
    random_state=42,
    alpha=0.0,
    tol=1e-8,
)
mlp.fit(X, y)

pred = mlp.predict(X)
ss_res = np.sum((y - pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1.0 - ss_res / ss_tot

result = {
    "coefficients": {
        "r_squared": float(r2),
    },
    "standard_errors": {
        "r_squared": 0.0,
    },
}

print(json.dumps(result, indent=2))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(42)

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["x1", "x2"]].values
y = df["y"].values

model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)
model.fit(X, y)

pred = model.predict(X)
mse = float(np.mean((y - pred) ** 2))
r2 = float(model.score(X, y))

result = {
    "coefficients": {
        "mse": mse,
        "r_squared": r2,
    },
    "standard_errors": {
        "mse": float("nan"),
        "r_squared": float("nan"),
    },
}

print(json.dumps(result, indent=2))

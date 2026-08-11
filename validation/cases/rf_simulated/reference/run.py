import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["x1", "x2"]].values
y = df["y"].values

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=None,
    bootstrap=True,
    n_jobs=-1,
)
model.fit(X, y)

result = {
    "r_squared": float(model.score(X, y)),
}

print(json.dumps(result, indent=2))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from quantile_forest import RandomForestQuantileRegressor

np.random.seed(42)

case_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(case_dir / "data" / "data.csv")

X = df[["x1", "x2"]].values
y = df["y"].values

model = RandomForestQuantileRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=42,
)
model.fit(X, y)

pred = model.predict(X, quantiles=0.75)
tss = np.sum((y - np.mean(y)) ** 2)
ss = np.sum((y - pred) ** 2)
r2 = float(1.0 - ss / tss)

result = {
    "coefficients": {"r_squared": r2},
    "standard_errors": {"r_squared": float("nan")},
}

print(json.dumps(result, indent=2))

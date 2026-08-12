import json
import numpy as np
import pandas as pd
from pathlib import Path
from econml.grf import CausalForest

np.random.seed(42)

case_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(case_dir / "data" / "data.csv")

X = df[["x1", "x2"]].values
T = df["treated"].values
y = df["y"].values

model = CausalForest(n_estimators=200, max_depth=4, random_state=42)
model.fit(X, T, y)

cate = model.predict(X)
ate = float(np.mean(cate))
se = float(np.std(cate, ddof=1) / np.sqrt(len(cate)))

result = {
    "coefficients": {"ate": ate},
    "standard_errors": {"ate": se},
}

print(json.dumps(result, indent=2))

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from econml.orf import DROrthoForest

warnings.filterwarnings("ignore")
np.random.seed(42)

case_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(case_dir / "data" / "data.csv")

X = df[["x1", "x2"]].values
W = df[["w1", "w2"]].values
T = df["treated"].values
y = df["y"].values

model = DROrthoForest(n_trees=50, max_depth=4, random_state=42, verbose=0)
model.fit(y, T, X=X, W=W)

effect = model.effect(X)
ate = float(np.mean(effect))
se = float(np.std(effect, ddof=1) / np.sqrt(len(effect)))

result = {
    "coefficients": {"ate": ate},
    "standard_errors": {"ate": se},
}

print(json.dumps(result, indent=2))

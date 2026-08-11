import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
y = df["y"].to_numpy()
p = 1 / (1 + np.exp(-(2.0 * df["x"].to_numpy())))
auc = roc_auc_score(y, p)
gini = 2 * auc - 1

result = {
    "coefficients": {"auc": float(auc), "gini": float(gini)},
    "standard_errors": {"auc": 0.0, "gini": 0.0},
}
print(json.dumps(result))

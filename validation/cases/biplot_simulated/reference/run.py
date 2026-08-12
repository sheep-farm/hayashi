import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["x1", "x2", "x3"]].values
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ev = S ** 2 / np.sum(S ** 2)
loadings = Vt.T * (S / np.sqrt(X.shape[0]))[:, np.newaxis]
s1 = float(np.sum(loadings[:, 0]))
s2 = float(np.sum(loadings[:, 1]))
result = {
  "coefficients": {
    "ev1": float(ev[0]),
    "ev2": float(ev[1]),
    "sum_loadings_pc1_sq": s1 * s1,
    "sum_loadings_pc2_sq": s2 * s2
  },
  "standard_errors": {"ev1": float("nan"), "ev2": float("nan"), "sum_loadings_pc1_sq": float("nan"), "sum_loadings_pc2_sq": float("nan")}
}
print(json.dumps(result))

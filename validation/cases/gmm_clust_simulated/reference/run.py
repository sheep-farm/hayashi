import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
gmm = GaussianMixture(n_components=2, random_state=42, n_init=1, max_iter=200, tol=1e-6).fit(df[["x1", "x2"]].values)
means = gmm.means_
order = np.argsort(means[:, 0])
means = means[order]
result = {
  "coefficients": {
    "mean_x_min": float(means[0, 0]),
    "mean_x_max": float(means[1, 0]),
    "mean_y_min": float(means[0, 1]),
    "mean_y_max": float(means[1, 1])
  },
  "standard_errors": {"mean_x_min": float("nan"), "mean_x_max": float("nan"), "mean_y_min": float("nan"), "mean_y_max": float("nan")}
}
print(json.dumps(result))

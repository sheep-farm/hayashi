import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
labels = DBSCAN(eps=0.5, min_samples=5).fit(df[["x1", "x2"]]).labels_
n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
n_noise = int(np.sum(labels == -1))
result = {
  "coefficients": {"n_clusters": float(n_clusters), "n_noise": float(n_noise)},
  "standard_errors": {"n_clusters": float("nan"), "n_noise": float("nan")}
}
print(json.dumps(result))

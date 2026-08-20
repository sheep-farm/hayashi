import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["x1", "x2"]].values
Z = linkage(X, method="ward")
labels = fcluster(Z, t=3, criterion="maxclust")
sizes = np.bincount(labels)[1:]
result = {
  "coefficients": {
    "n_clusters": float(3),
    "min_cluster_size": float(min(sizes)),
    "max_cluster_size": float(max(sizes))
  },
  "standard_errors": {
    "n_clusters": float("nan"),
    "min_cluster_size": float("nan"),
    "max_cluster_size": float("nan")
  }
}
print(json.dumps(result))

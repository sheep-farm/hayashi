import json
import numpy as np
import pandas as pd
from pathlib import Path
from umap import UMAP
from sklearn.cluster import KMeans

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["x1", "x2", "x3"]].values
emb = UMAP(n_components=2, n_neighbors=15, n_epochs=200, random_state=42).fit_transform(X)
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(emb)
result = {
  "coefficients": {"n_clusters": float(km.n_clusters)},
  "standard_errors": {"n_clusters": float("nan")}
}
print(json.dumps(result))

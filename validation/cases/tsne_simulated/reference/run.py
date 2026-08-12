import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data.csv")
X = df[["x1", "x2", "x3"]].values
emb = TSNE(n_components=2, perplexity=10, max_iter=250, learning_rate=100, random_state=42, init="random").fit_transform(X)
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(emb)
result = {
  "coefficients": {"n_clusters": float(km.n_clusters)},
  "standard_errors": {"n_clusters": float("nan")}
}
print(json.dumps(result))

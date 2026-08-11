import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 200
n_panels = 50
T = 4

ids = np.repeat(np.arange(n_panels), T)
x = np.random.normal(0, 1, n)
beta = np.array([1.0, 0.5])  # intercept, x
alpha = np.repeat(np.random.normal(0, 0.5, n_panels), T)
eps = np.random.normal(0, 0.5, n)

y_latent = beta[0] + beta[1] * x + alpha + eps
y = np.maximum(0.0, y_latent)

df = pd.DataFrame({"id": ids, "x": x, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

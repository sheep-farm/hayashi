import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 1000
n_panels = 100
t = n // n_panels
ids = np.repeat(np.arange(1, n_panels + 1), t)[:n]
x = np.random.uniform(0, 1, n)
alpha = np.random.normal(0, 1.0, n_panels)
entity_effect = np.repeat(alpha, t)[:n]
# Heteroskedastic error increasing with x
u = np.random.exponential(1.0, n) - 1.0
eps = (1.0 + 0.5 * x) * u
y = 1.0 + 2.0 * x + entity_effect + eps

df = pd.DataFrame({"id": ids, "x": x, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

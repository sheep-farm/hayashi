import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_panels = 50
t = 100
n = n_panels * t
ids = np.repeat(np.arange(1, n_panels + 1), t)
y1 = np.zeros(n)
y2 = np.zeros(n)
for i in range(n_panels):
    idx = i * t
    y1[idx] = np.random.normal(0, 1)
    y2[idx] = np.random.normal(0, 1)
    for j in range(1, t):
        y1[idx + j] = 0.3 * y1[idx + j - 1] + 0.2 * y2[idx + j - 1] + np.random.normal(0, 0.5)
        y2[idx + j] = 0.1 * y1[idx + j - 1] + 0.4 * y2[idx + j - 1] + np.random.normal(0, 0.5)

df = pd.DataFrame({"id": ids, "y1": y1, "y2": y2})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

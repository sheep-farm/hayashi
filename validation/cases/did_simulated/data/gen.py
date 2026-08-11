import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 100
treat = (np.random.uniform(0, 1, n) > 0.5).astype(int)
pre = np.random.normal(2, 0.5, n)
att = 1.5
y0 = pre + np.random.normal(0, 0.2, n)
y1 = pre + att * treat + np.random.normal(0, 0.2, n) + 0.2  # common trend

# Long format
df = pd.DataFrame({
    "id": np.tile(np.arange(n), 2),
    "post": [0] * n + [1] * n,
    "treat": np.tile(treat, 2),
    "y": np.concatenate([y0, y1])
})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

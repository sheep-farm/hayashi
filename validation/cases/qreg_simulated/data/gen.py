import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 500
x = np.random.uniform(0, 1, n)
b0 = 1.0
b1 = 2.0
eps = np.random.exponential(1.0, n) - 1.0  # mean 0, skewed
y = b0 + b1 * x + (1.0 + 0.5 * x) * eps

df = pd.DataFrame({"x": x, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

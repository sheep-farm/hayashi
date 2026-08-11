import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 500
y = np.zeros(n)
y[0] = 0.0
for t in range(1, n):
    if y[t - 1] < 0.0:
        y[t] = 0.50 + 0.80 * y[t - 1] + np.random.normal(0, 0.3)
    else:
        y[t] = -0.50 + 0.30 * y[t - 1] + np.random.normal(0, 0.3)

df = pd.DataFrame({"y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

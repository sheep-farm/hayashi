import os
import numpy as np
import pandas as pd

np.random.seed(40)
n = 500

q = np.random.uniform(-1.0, 1.0, n)

y1 = np.zeros(n)
y2 = np.zeros(n)
for t in range(1, n):
    if q[t - 1] < 0.0:
        y1[t] = 0.2 + 0.3 * y1[t - 1] + 0.1 * y2[t - 1] + np.random.normal(0.0, 0.3)
        y2[t] = -0.1 + 0.2 * y1[t - 1] + 0.4 * y2[t - 1] + np.random.normal(0.0, 0.3)
    else:
        y1[t] = 0.4 + 0.1 * y1[t - 1] - 0.2 * y2[t - 1] + np.random.normal(0.0, 0.3)
        y2[t] = 0.3 - 0.1 * y1[t - 1] + 0.5 * y2[t - 1] + np.random.normal(0.0, 0.3)

pd.DataFrame({"y1": y1, "y2": y2, "q": q}).to_csv(
    os.path.join(os.path.dirname(__file__), "data.csv"), index=False
)

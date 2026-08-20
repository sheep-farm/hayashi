import numpy as np
import pandas as pd
import os

np.random.seed(789)
n = 250
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
y = 3.0 * x1 + 2.0 * x2 + np.random.normal(0, 0.1, n)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

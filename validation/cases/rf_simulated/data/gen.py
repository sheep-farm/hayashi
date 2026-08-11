import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
y = 3.0 * x1 + np.random.normal(0, 0.1, n)

df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

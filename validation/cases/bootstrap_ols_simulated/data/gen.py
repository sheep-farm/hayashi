import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 100
x = np.random.normal(0, 1, n)
y = 1.0 + 2.0 * x + np.random.normal(0, 0.5, n)

df = pd.DataFrame({"y": y, "x": x})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

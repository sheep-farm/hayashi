import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

x = np.cumsum(np.random.normal(0.0, 0.1, n))
e = np.random.normal(0.0, 0.5, n)
y = 0.5 * x + e

pd.DataFrame({"y": y, "x": x}).to_csv(
    os.path.join(os.path.dirname(__file__), "data.csv"), index=False
)

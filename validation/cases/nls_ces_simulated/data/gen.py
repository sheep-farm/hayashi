import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 100
x1 = np.random.uniform(0.5, 5.0, n)
x2 = np.random.uniform(0.5, 5.0, n)
a = 1.2
b1 = 0.4
rho = 0.5
y = a * (b1 * (x1 ** rho) + (1 - b1) * (x2 ** rho)) ** (1.0 / rho) + np.random.normal(0, 0.1, n)

df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

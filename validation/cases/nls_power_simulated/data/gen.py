import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 100
x = np.linspace(0.5, 5.0, n)
a = 2.0
b = 0.8
y = a * (x ** b) + np.random.normal(0, 0.3, n)

df = pd.DataFrame({"x": x, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

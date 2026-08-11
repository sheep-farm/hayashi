import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 100
x = np.linspace(0.0, 5.0, n)
a = 10.0
b = 2.0
c = 2.5
y = a / (1.0 + np.exp(-b * (x - c))) + np.random.normal(0, 0.2, n)

df = pd.DataFrame({"x": x, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

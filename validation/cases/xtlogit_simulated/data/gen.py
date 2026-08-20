import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 50
t = 4
N = n * t

id = np.repeat(np.arange(n), t)
x = np.random.normal(0, 1, N)
b0 = -0.5
b1 = 0.8
lp = b0 + b1 * x
p = 1.0 / (1.0 + np.exp(-lp))
y = np.random.binomial(1, p)

df = pd.DataFrame({"id": id, "x": x, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

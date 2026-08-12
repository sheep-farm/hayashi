import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

x1 = np.exp(np.random.normal(0.0, 0.4, n))
x2 = np.exp(np.random.normal(0.0, 0.4, n))
lx1 = np.log(x1)
lx2 = np.log(x2)

b0, b1, b2 = 1.0, 0.4, 0.5
v = np.random.normal(0.0, 0.15, n)
ly = b0 + b1 * lx1 + b2 * lx2 + v
y = np.exp(ly)

df = pd.DataFrame({
    "y": y,
    "x1": x1,
    "x2": x2,
    "ly": ly,
    "lx1": lx1,
    "lx2": lx2,
})

df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

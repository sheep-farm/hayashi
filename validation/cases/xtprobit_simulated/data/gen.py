import numpy as np
import pandas as pd
import os
from scipy.special import erf

np.random.seed(42)
n = 50
t = 4
N = n * t

id = np.repeat(np.arange(n), t)
x = np.random.normal(0, 1, N)
b0 = -0.3
b1 = 0.6
lp = b0 + b1 * x
p = 0.5 * (1.0 + erf(lp / np.sqrt(2)))
y = (np.random.uniform(0, 1, N) < p).astype(int)

df = pd.DataFrame({"id": id, "x": x, "y": y})
df.to_csv(os.path.join(os.path.dirname(__file__), "data.csv"), index=False)

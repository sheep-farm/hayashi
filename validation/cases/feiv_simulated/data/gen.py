import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 200
T = 5
n = N * T
ids = np.repeat(np.arange(N), T)
time = np.tile(np.arange(T), N)
z = np.random.normal(0, 1, n)
u = np.random.normal(0, 1, n)
alpha = np.repeat(np.random.normal(0, 1, N), T)
x = 0.5 * z + 0.3 * u + alpha + np.random.normal(0, 0.5, n)
y = 0.8 * x + alpha + u + np.random.normal(0, 0.5, n)

df = pd.DataFrame({"id": ids, "t": time, "y": y, "x": x, "z": z})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

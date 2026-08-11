import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 50
T = 4
n = N * T
ids = np.repeat(np.arange(N), T)
time = np.tile(np.arange(T), N)
alpha = np.repeat(np.random.normal(0, 1, N), T)
x = np.random.normal(0, 1, n)
y = 0.6 * x + alpha + np.random.normal(0, 0.5, n)

df = pd.DataFrame({"id": ids, "t": time, "y": y, "x": x})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 300
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)
eps = np.random.normal(0, 0.1 * (1 + 0.5 * x1), n)
y = 3.0 * x1 + eps

df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

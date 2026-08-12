import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 500
x = np.random.normal(0, 1, n)
y0 = 1.0 + 0.5 * x + np.random.normal(0, 0.5, n)
ate = 2.0
ps = 1.0 / (1.0 + np.exp(-(0.3 * x)))
d = (np.random.uniform(0, 1, n) < ps).astype(int)
y = y0 + d * ate + np.random.normal(0, 0.2, n)

df = pd.DataFrame({"y": y, "d": d, "x": x})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

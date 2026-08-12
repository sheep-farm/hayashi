import json
import numpy as np
import os
import pandas as pd

np.random.seed(123)

n = 100
y1 = np.zeros(n)
y2 = np.zeros(n)
e1 = np.random.normal(scale=0.5, size=n)
e2 = np.random.normal(scale=0.5, size=n)

y1[0] = e1[0]
y2[0] = e2[0]

for t in range(1, n):
    y1[t] = 0.20 * y1[t - 1] + 0.30 * y2[t - 1] + e1[t]
    y2[t] = 0.10 * y1[t - 1] + 0.40 * y2[t - 1] + e2[t]

outdir = os.path.dirname(__file__)
df = pd.DataFrame({"y1": y1, "y2": y2})
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

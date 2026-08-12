import json
import numpy as np
import os
import pandas as pd

np.random.seed(42)

n = 150
f1 = np.random.normal(size=n)
f2 = np.random.normal(size=n)

eps1 = np.random.normal(scale=0.3, size=n)
eps2 = np.random.normal(scale=0.4, size=n)
eps3 = np.random.normal(scale=0.3, size=n)
eps4 = np.random.normal(scale=0.5, size=n)
eps5 = np.random.normal(scale=0.4, size=n)

y1 = 0.7 * f1 + 0.2 * f2 + eps1
y2 = 0.6 * f1 - 0.1 * f2 + eps2
y3 = -0.5 * f1 + 0.3 * f2 + eps3
y4 = 0.2 * f1 + 0.7 * f2 + eps4
y5 = 0.4 * f1 + 0.4 * f2 + eps5

outdir = os.path.dirname(__file__)
df = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3, "y4": y4, "y5": y5})

# Standardise to unit population variance so that residual variances are
# directly interpretable as uniquenesses.
mu = df.mean()
sigma = df.std(ddof=0)
df_std = (df - mu) / sigma

df_std.to_csv(os.path.join(outdir, "data.csv"), index=False)

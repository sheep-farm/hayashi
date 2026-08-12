import json
import numpy as np
import os
import pandas as pd

np.random.seed(42)

n = 300
f = np.random.normal(size=n)
y1 = 0.5 * f + np.random.normal(scale=0.3, size=n)
y2 = 1.0 * f + np.random.normal(scale=0.3, size=n)
y3 = -0.8 * f + np.random.normal(scale=0.3, size=n)

outdir = os.path.dirname(__file__)
df = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

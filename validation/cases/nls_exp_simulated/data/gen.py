import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 100
x = np.linspace(0.1, 2.0, n)
a = 2.5
b = -0.8
y = a * np.exp(b * x) + np.random.normal(0, 0.1, n)

df = pd.DataFrame({"x": x, "y": y})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

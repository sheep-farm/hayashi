import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
w1 = np.random.normal(0, 1, n)
w2 = np.random.normal(0, 1, n)
treated = (np.random.rand(n) > 0.5).astype(int)
# Constant additive treatment effect of 0.5; w1 and w2 are additional confounders
y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * treated + 0.3 * w1 - 0.2 * w2 + np.random.normal(0, 1, n)

df = pd.DataFrame({"x1": x1, "x2": x2, "w1": w1, "w2": w2, "treated": treated, "y": y})
outdir = os.path.dirname(__file__)
df.to_csv(os.path.join(outdir, "data.csv"), index=False)

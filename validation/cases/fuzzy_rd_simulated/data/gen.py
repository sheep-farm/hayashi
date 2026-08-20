import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)
n = 1000
x = np.random.normal(0, 1.5, n)
Z = (x >= 0).astype(float)
comp = np.random.binomial(1, 0.7, n)
noise = np.random.rand(n) * 1e-6
d = np.where((Z > 0) & (comp == 1), 1.0 + noise, 0.0)
y = 0.5 + 2.0 * (d > 0).astype(float) + 1.5 * x + np.random.normal(0, 0.5, n)
df = pd.DataFrame({"y": y, "x": x, "d": d, "Z": Z})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

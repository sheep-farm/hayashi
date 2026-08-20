import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(47)
n = 200
z = np.random.normal(0, 1, n)
u = np.random.normal(0, 1, n)
v = 0.6 * u + np.random.normal(0, np.sqrt(1 - 0.36), n)
x = 0.5 + 0.7 * z + v
y = 1.0 + 0.8 * x + u + np.random.normal(0, 0.5, n)
df = pd.DataFrame({"y": y, "x": x, "z": z})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

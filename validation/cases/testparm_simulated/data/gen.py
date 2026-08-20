import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(50)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
y = 1.0 + 0.5 * x1 + 0.5 * x2 + np.random.normal(0, 1, n)
df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

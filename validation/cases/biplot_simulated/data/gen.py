import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)
n = 100
x1 = np.random.uniform(0, 10, n)
x2 = x1 + np.random.normal(0, 0.5, n)
x3 = x1 + np.random.normal(0, 0.5, n)
df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

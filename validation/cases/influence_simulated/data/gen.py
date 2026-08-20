import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(49)
n = 200
x = np.random.normal(0, 1, n)
y = 1.0 + 2.0 * x + np.random.normal(0, 1, n)
# add a strong outlier
x[0] = 5.0
y[0] = 30.0
df = pd.DataFrame({"y": y, "x": x})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

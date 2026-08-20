import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)
n = 200
x = np.random.normal(0, 1, n)
logit_p = 2.0 * x
p = 1 / (1 + np.exp(-logit_p))
y = (np.random.uniform(0, 1, n) < p).astype(int)
df = pd.DataFrame({"y": y, "x": x})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

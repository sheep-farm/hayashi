import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)
n = 400
pre = np.random.normal(5, 2, n)
t = np.random.binomial(1, 0.5, n).astype(float)
y = 0.5 + 2.0 * t + 0.8 * pre + np.random.normal(0, 0.5, n)
df = pd.DataFrame({"y": y, "pre_y": pre, "treated": t})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

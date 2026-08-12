import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 1000
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
ps = 1.0 / (1.0 + np.exp(-(0.5 * x1 - 0.3 * x2)))
t = (np.random.uniform(size=n) < ps).astype(int)
mu = 1.0 + 0.8 * x1 + 0.4 * x2 + 0.7 * t
y = mu + np.random.normal(0.0, 1.0, size=n)

df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
df.to_csv(DATA_DIR / "data.csv", index=False)

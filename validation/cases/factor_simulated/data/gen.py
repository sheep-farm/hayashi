import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 300
f1 = np.random.normal(size=n)
f2 = np.random.normal(size=n)
# 4 observed variables, each loads mainly on one factor + small cross-loading
x1 = 1.0 * f1 + 0.2 * f2 + np.random.normal(0, 0.4, size=n)
x2 = 0.9 * f1 + 0.3 * f2 + np.random.normal(0, 0.4, size=n)
x3 = 0.2 * f1 + 1.0 * f2 + np.random.normal(0, 0.4, size=n)
x4 = 0.3 * f1 + 0.9 * f2 + np.random.normal(0, 0.4, size=n)

df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
df.to_csv(DATA_DIR / "data.csv", index=False)

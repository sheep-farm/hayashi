import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 150
group = np.repeat([1, 2, 3], n // 3)
# DGP: group 1 -> y1=0, y2=5; group 2 -> y1=1, y2=3; group 3 -> y1=2, y2=1
mu = np.array([[0.0, 5.0], [1.0, 3.0], [2.0, 1.0]])
y1 = np.zeros(n)
y2 = np.zeros(n)
for i, g in enumerate(group):
    y1[i] = mu[g - 1, 0] + np.random.normal(0, 0.3)
    y2[i] = mu[g - 1, 1] + np.random.normal(0, 0.3)

df = pd.DataFrame({"y1": y1, "y2": y2, "group": group})
df.to_csv(DATA_DIR / "data.csv", index=False)

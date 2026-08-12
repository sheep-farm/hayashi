import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 500
mu = -1.0
phi = 0.95
sigma = 0.2

h = np.zeros(n)
for t in range(1, n):
    h[t] = mu + phi * (h[t - 1] - mu) + sigma * np.random.normal()

y = np.exp(h / 2) * np.random.normal(size=n)

df = pd.DataFrame({"y": y})
df.to_csv(DATA_DIR / "data.csv", index=False)

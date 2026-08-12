import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 80
x = np.sort(np.random.uniform(0, 1, n))
y = np.sin(4 * x) + 0.2 * np.random.normal(size=n)

df = pd.DataFrame({"x": x, "y": y})
df.to_csv(DATA_DIR / "data.csv", index=False)

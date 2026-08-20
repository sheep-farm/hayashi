import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 300
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
y = 1.0 + 2.0 * x1 - 1.5 * x2 + 0.5 * np.random.normal(size=n)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
df.to_csv(DATA_DIR / "data.csv", index=False)

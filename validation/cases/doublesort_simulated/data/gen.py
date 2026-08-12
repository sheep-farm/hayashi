import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 1000
size = np.random.normal(size=n)
bm = np.random.normal(size=n)
# ret is higher for small/high-BM (value) firms
ret = 0.02 - 0.01 * size + 0.015 * bm + 0.05 * np.random.normal(size=n)

df = pd.DataFrame({"ret": ret, "size": size, "bm": bm})
df.to_csv(DATA_DIR / "data.csv", index=False)

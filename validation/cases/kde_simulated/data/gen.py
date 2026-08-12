import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 500
x = np.random.normal(loc=2.0, scale=1.5, size=n)

df = pd.DataFrame({"x": x})
df.to_csv(DATA_DIR / "data.csv", index=False)

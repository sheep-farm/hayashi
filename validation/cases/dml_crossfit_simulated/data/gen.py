import numpy as np
import pandas as pd
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 600
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
g = 0.5 * x1 + 0.3 * x2
# treatment is correlated with x
prob = 1.0 / (1.0 + np.exp(-(0.4 * x1 - 0.3 * x2)))
d = (np.random.uniform(size=n) < prob).astype(float)
theta = 1.5
# shift d by 0.5 to avoid Hayashi bool inference on 0/1 columns
y = theta * d + g + 0.5 * np.random.normal(size=n)
dvar = d + 0.5

df = pd.DataFrame({"y": y, "dvar": dvar, "x1": x1, "x2": x2})
df.to_csv(DATA_DIR / "data.csv", index=False)

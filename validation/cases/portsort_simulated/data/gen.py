import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)
n = 200
size = np.random.uniform(0, 100, n)
ret = 0.005 + 0.0001 * size + np.random.normal(0, 0.02, n)
df = pd.DataFrame({"size": size, "ret": ret})
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

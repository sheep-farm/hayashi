import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)

def blob(n, c, s):
    return np.random.normal(c, s, (n, 2))

pts = np.vstack([blob(50, [0.0, 0.0], 0.1),
                 blob(50, [3.0, 3.0], 0.1),
                 blob(50, [6.0, 0.0], 0.1),
                 np.array([[10.0, 10.0], [10.0, -10.0], [-10.0, 10.0], [-10.0, -10.0], [0.0, 10.0]])])
df = pd.DataFrame(pts, columns=["x1", "x2"])
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

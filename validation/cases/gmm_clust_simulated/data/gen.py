import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)
pts = []
pts += [np.random.normal([0.0, 0.0], 0.4, 2) for _ in range(100)]
pts += [np.random.normal([5.0, 5.0], 0.4, 2) for _ in range(100)]
df = pd.DataFrame(np.array(pts), columns=["x1", "x2"])
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

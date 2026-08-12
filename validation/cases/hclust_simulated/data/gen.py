import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)
pts = []
for c in [[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]]:
    pts += [c + np.random.normal(0, 0.2, 2) for _ in range(20)]
df = pd.DataFrame(np.array(pts), columns=["x1", "x2"])
df.to_csv(Path(__file__).resolve().parent / "data.csv", index=False)

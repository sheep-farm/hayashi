"""Generate two blocks of correlated variables for canonical correlation."""

from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(42)
n_obs = 200

x1 = rng.normal(0, 1, n_obs)
x2 = 0.5 * x1 + rng.normal(0, 0.5, n_obs)
y1 = 0.8 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n_obs)
y2 = -0.5 * x1 + 0.6 * x2 + rng.normal(0, 0.4, n_obs)

pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2}).to_csv(
    Path(__file__).resolve().parent / "data.csv", index=False
)

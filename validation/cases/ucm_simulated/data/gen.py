"""Generate a deterministic local-linear-trend + seasonal series."""

from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(42)
n_obs = 36
time = np.arange(1, n_obs + 1)

level = 1.0 + 0.05 * time
slope = 0.05
seasonal = 2.0 * np.sin(2 * np.pi * time / 12)
noise = rng.normal(0, 0.2, n_obs)
y = level + seasonal + noise

pd.DataFrame({"t": time, "y": y}).to_csv(
    Path(__file__).resolve().parent / "data.csv", index=False
)

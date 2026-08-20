"""Generate deterministic series for seasonal decomposition validation."""

from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(42)
n_obs = 120
time = np.arange(1, n_obs + 1)

# linear trend + 12-month seasonal dummies + small noise
trend = 0.05 * time
seasonal = 2.0 * np.sin(2 * np.pi * time / 12) + 1.5 * np.cos(2 * np.pi * time / 12)
noise = rng.normal(0, 0.1, n_obs)
y = 1.0 + trend + seasonal + noise

pd.DataFrame({"t": time, "y": y}).to_csv(
    Path(__file__).resolve().parent / "data.csv", index=False
)

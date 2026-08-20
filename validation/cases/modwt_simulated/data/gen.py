"""Generate the deterministic MODWT validation series."""

from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(42)
n_obs = 128
time = np.arange(n_obs)
series = 0.02 * time + np.sin(2 * np.pi * time / 16) + rng.normal(0, 0.1, n_obs)

pd.DataFrame({"y": series}).to_csv(
    Path(__file__).resolve().parent / "data.csv", index=False
)

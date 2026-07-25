# Data generation for the synthetic control smoking case.

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "synth_smoking.csv"

np.random.seed(42)

if not CSV_PATH.exists():
    n_units = 10
    n_periods = 20
    unit = np.repeat(np.arange(1, n_units + 1), n_periods)
    year = np.tile(np.arange(1, n_periods + 1), n_units)
    alpha = np.where(unit == 1, 5.0, unit * 1.0)
    d = (unit == 1) * (year >= 11)
    e = np.random.normal(size=len(unit))
    y = alpha + 0.3 * year + 3.0 * d + e
    df = pd.DataFrame({"unit": unit, "year": year, "y": y, "alpha": alpha, "d": d})
    df.to_csv(CSV_PATH, index=False)

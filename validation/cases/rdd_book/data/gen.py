# Data generation for the RDD book case.

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "rdd_book.csv"

np.random.seed(42)

if not CSV_PATH.exists():
    n = 1000
    x = np.random.uniform(-1.0, 1.0, size=n)
    d = (x >= 0.0).astype(float)
    e = np.random.normal(size=n)
    y = 1.0 + 0.5 * x + 2.0 * d + e
    df = pd.DataFrame({"x": x, "y": y, "d": d})
    df.to_csv(CSV_PATH, index=False)

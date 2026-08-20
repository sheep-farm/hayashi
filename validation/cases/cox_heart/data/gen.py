# Data generation for the Cox survival heart case.

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "heart.csv"

if not CSV_PATH.exists():
    heart = sm.datasets.heart.load_pandas().data
    heart = heart.rename(columns={"censors": "censored", "survival": "time"})
    heart.to_csv(CSV_PATH, index=False)

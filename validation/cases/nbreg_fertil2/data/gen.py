# Data generation for the Negative Binomial fertil2 case.

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "fertil2.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("fertil2")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/fertil2.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)

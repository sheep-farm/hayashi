# Data generation for the GLSAR housing price case.

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "hprice1.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("hprice1")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/hprice1.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)

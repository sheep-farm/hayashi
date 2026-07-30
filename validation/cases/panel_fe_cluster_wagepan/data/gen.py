# Data generation for the panel FE clustered-SE wagepan case.

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "wagepan.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data

        df = data("wagepan")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/wagepan.csv"
        df = pd.read_csv(url)

    variables = [
        "lwage",
        "union",
        "married",
        "d81",
        "d82",
        "d83",
        "d84",
        "d85",
        "d86",
        "d87",
        "nr",
        "year",
    ]
    df = df[variables].dropna()
    df.to_csv(CSV_PATH, index=False)

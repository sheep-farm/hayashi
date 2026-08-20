from pathlib import Path

import numpy as np
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
        if "rownames" in df.columns:
            df = df.drop(columns=["rownames"])

    if "lprice" not in df.columns:
        df["lprice"] = np.log(df["price"])
    if "llotsize" not in df.columns:
        df["llotsize"] = np.log(df["lotsize"])
    if "lsqrft" not in df.columns:
        df["lsqrft"] = np.log(df["sqrft"])

    df.to_csv(CSV_PATH, index=False)

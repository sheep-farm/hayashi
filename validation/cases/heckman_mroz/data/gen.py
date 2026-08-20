# Data generation for the Heckman two-step selection model case.

from pathlib import Path

from wooldridge import data

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "mroz.csv"

if not CSV_PATH.exists():
    df = data("mroz")
    needed = ["inlf", "lwage", "educ", "exper", "expersq", "age",
              "kidslt6", "kidsge6", "nwifeinc"]
    df = df[needed].copy()
    df["lwage"] = df["lwage"].fillna(0.0)
    df.to_csv(CSV_PATH, index=False)

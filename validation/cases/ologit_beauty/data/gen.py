# Data generation for the ordered logit beauty case.

from pathlib import Path

from wooldridge import data

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "beauty.csv"

if not CSV_PATH.exists():
    df = data("beauty")
    df = df[df["looks"].isin([2, 3, 4])].copy()
    df.to_csv(CSV_PATH, index=False)

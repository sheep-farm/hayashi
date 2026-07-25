# Data generation for the multinomial logit mode-choice case.

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = DATA_DIR / "TravelMode.csv"
CSV_PATH = DATA_DIR / "mode.csv"

if not RAW_CSV.exists():
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/TravelMode.csv"
    pd.read_csv(url).to_csv(RAW_CSV, index=False)

raw = pd.read_csv(RAW_CSV)

avg = raw.groupby("individual")[["wait", "vcost", "travel"]].mean().reset_index()
chosen = raw[raw["choice"] == "yes"][["individual", "mode", "income"]].copy()
chosen = chosen.merge(avg, on="individual")

mode_map = {"air": 1, "train": 2, "bus": 3, "car": 4}
chosen["mode"] = chosen["mode"].map(mode_map)
chosen = chosen.drop(columns=["individual"])

for col in ["income", "wait", "vcost", "travel"]:
    chosen[col] = (chosen[col] - chosen[col].mean()) / chosen[col].std()

chosen.to_csv(CSV_PATH, index=False)

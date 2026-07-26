# Reference implementation in Python/NumPy for the Wooldridge wage1 summary case.

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "wage1.csv"

# Load the same CSV that Hayashi will read, or generate it if absent.
if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("wage1")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/wage1.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

x = df["wage"].dropna().to_numpy()
n = int(len(x))
mean_x = float(np.mean(x))
sd_x = float(np.std(x, ddof=1))
min_x = float(np.min(x))
max_x = float(np.max(x))

# Quantiles using nearest-rank rounding to match Hayashi.
ordered = np.sort(x)
idx25 = int(round(0.25 * (n - 1)))
idx50 = int(round(0.50 * (n - 1)))
idx75 = int(round(0.75 * (n - 1)))
p25 = float(ordered[idx25])
p50 = float(ordered[idx50])
p75 = float(ordered[idx75])

# Skewness and kurtosis use the same moment formulas as Hayashi.
skew = float(np.sum(((x - mean_x) / sd_x) ** 3) * n / ((n - 1) * (n - 2)))
kurt = float(np.mean(((x - mean_x) / sd_x) ** 4))

result = {
    "N": n,
    "mean": mean_x,
    "sd": sd_x,
    "min": min_x,
    "max": max_x,
    "p25": p25,
    "p50": p50,
    "p75": p75,
    "skewness": skew,
    "kurtosis": kurt,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

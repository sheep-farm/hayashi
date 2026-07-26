# Reference implementation in Python for the Wooldridge wage1 codebook case.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "wage.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("wage1")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/wage1.csv"
        df = pd.read_csv(url)
    df[["wage"]].to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

x = df["wage"].dropna().to_numpy()
n = int(len(x))
ordered = np.sort(x)
idx25 = int(round(0.25 * (n - 1)))
idx50 = int(round(0.50 * (n - 1)))
idx75 = int(round(0.75 * (n - 1)))

result = {
    "variable": ["wage"],
    "type": ["float"],
    "obs": [n],
    "missing": [int(df["wage"].isna().sum())],
    "unique": [int(len(np.unique(x)))],
    "mean": [float(np.mean(x))],
    "sd": [float(np.std(x, ddof=1))],
    "min": [float(np.min(x))],
    "p25": [float(ordered[idx25])],
    "p50": [float(ordered[idx50])],
    "p75": [float(ordered[idx75])],
    "max": [float(np.max(x))],
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

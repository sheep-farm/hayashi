# Reference implementation in Python for the Wooldridge wage1 centile case.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "wage1.csv"

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
n = len(x)
ordered = np.sort(x)

pcts = [10, 25, 50, 75, 90]
# Hayashi uses round(p/100 * (n-1)) with half away from zero.
idx = [int(p / 100 * (n - 1) + 0.5) for p in pcts]
values = [float(ordered[i]) for i in idx]

result = {
    "centile": pcts,
    "value": values,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

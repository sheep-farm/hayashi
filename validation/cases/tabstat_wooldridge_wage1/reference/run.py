# Reference implementation in Python for the Wooldridge wage1 tabstat case.

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

vars = ["wage", "educ", "exper", "tenure"]
stats = ["mean", "sd", "min", "max", "p50"]

variable = []
stat = []
value = []

for v in vars:
    x = df[v].dropna().to_numpy()
    n = len(x)
    ordered = np.sort(x)
    idx50 = int(round(0.50 * (n - 1)))
    for s in stats:
        variable.append(v)
        stat.append(s)
        if s == "mean":
            value.append(float(np.mean(x)))
        elif s == "sd":
            value.append(float(np.std(x, ddof=1)))
        elif s == "min":
            value.append(float(np.min(x)))
        elif s == "max":
            value.append(float(np.max(x)))
        elif s == "p50":
            idx = int(0.50 * (n - 1) + 0.5)
            value.append(float(ordered[idx]))

result = {
    "variable": variable,
    "stat": stat,
    "value": value,
    "group": ["all"] * len(variable),
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

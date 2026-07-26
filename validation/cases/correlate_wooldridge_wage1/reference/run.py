# Reference implementation in Python for the Wooldridge wage1 correlation case.

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
sub = df[vars].dropna()
mat = sub.corr()
n = len(sub)

sorted_vars = sorted(vars)
var1 = []
var2 = []
r_vals = []
p_vals = []

for i in sorted_vars:
    for j in sorted_vars:
        if sorted_vars.index(j) > sorted_vars.index(i):
            continue
        rij = float(mat.loc[i, j])
        if i == j:
            pij = 0.0
        elif n <= 2 or (1 - rij**2) <= 0:
            pij = 1.0
        else:
            t = rij * math.sqrt((n - 2) / (1 - rij**2))
            pij = float(2 * (1 - stats.t.cdf(abs(t), n - 2)))
        var1.append(i)
        var2.append(j)
        r_vals.append(rij)
        p_vals.append(pij)

result = {
    "var1": var1,
    "var2": var2,
    "r": r_vals,
    "p": p_vals,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

# Reference implementation in Python for the Wooldridge mroz cross-tabulation case.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "mroz.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("mroz")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/mroz.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Ensure categorical string labels.
df["inlf"] = df["inlf"].astype(str)
df["kidslt6"] = df["kidslt6"].astype(str)

# Cross-tabulation.
tab = pd.crosstab(df["inlf"], df["kidslt6"])
chi2, p, dof, expected = chi2_contingency(tab, correction=False)

rows = [str(r) for r in tab.index]
cols = [str(c) for c in tab.columns]

var1 = []
var2 = []
freq = []
row_total = []
col_total = []

for r in rows:
    rt = int(tab.loc[r].sum())
    for c in cols:
        var1.append(r)
        var2.append(c)
        freq.append(int(tab.loc[r, c]))
        row_total.append(rt)
        col_total.append(int(tab[c].sum()))

result = {
    "chi2": float(chi2),
    "df": int(dof),
    "p_value": float(p),
    "table": {
        "inlf": var1,
        "kidslt6": var2,
        "freq": freq,
        "row_total": row_total,
        "col_total": col_total,
    },
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

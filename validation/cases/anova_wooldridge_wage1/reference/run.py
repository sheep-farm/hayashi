# Reference implementation in Python for the Wooldridge wage1 one-way ANOVA case.

import json
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

# One-way ANOVA of wage by educ.
groups = [g.dropna().to_numpy() for _, g in df.groupby("educ")["wage"]]
f_stat, p_value = stats.f_oneway(*groups)

n_total = len(df)
n_groups = len(groups)
df_between = n_groups - 1
df_within = n_total - n_groups

overall_mean = float(np.mean(df["wage"].dropna()))
ss_between = float(sum(len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups))
ss_within = float(sum(np.sum((g - np.mean(g)) ** 2) for g in groups))
ss_total = ss_between + ss_within
ms_between = ss_between / df_between
ms_within = ss_within / df_within

result = {
    "test": "One-Way ANOVA",
    "ss_between": ss_between,
    "ss_within": ss_within,
    "ss_total": ss_total,
    "df_between": df_between,
    "df_within": df_within,
    "ms_between": ms_between,
    "ms_within": ms_within,
    "f_stat": float(f_stat),
    "p_value": float(p_value),
    "n_groups": n_groups,
    "n_obs": n_total,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

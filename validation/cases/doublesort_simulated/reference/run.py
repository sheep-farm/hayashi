import json
import numpy as np
import pandas as pd

CASE_DIR = "validation/cases/doublesort_simulated"
DATA_DIR = f"{CASE_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/data.csv")
df["size_q"] = pd.qcut(df["size"], 5, labels=False, duplicates="drop") + 1
df["bm_q"] = pd.qcut(df["bm"], 5, labels=False, duplicates="drop") + 1

mean_ret = df.groupby(["size_q", "bm_q"])["ret"].mean()
val = mean_ret.loc[(1, 5)]

result = {
    "coefficients": {"low_size_high_bm": float(val)},
    "standard_errors": {"low_size_high_bm": 0.0},
}

print(json.dumps(result))

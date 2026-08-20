# Reference implementation in Python for the Wooldridge wagepan xtsum case.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "wagepan.csv"

if not CSV_PATH.exists():
    try:
        from wooldridge import data
        df = data("wagepan")
    except ImportError:
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/wagepan.csv"
        df = pd.read_csv(url)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

y = df["lwage"].to_numpy()
id_arr = df["nr"].to_numpy()

n_total = len(y)
n_entities = len(np.unique(id_arr))

# Overall.
overall_mean = float(np.mean(y))
overall_sd = float(np.std(y, ddof=1))
overall_min = float(np.min(y))
overall_max = float(np.max(y))

# Between: entity means.
df_calc = pd.DataFrame({"y": y, "id": id_arr})
entity_means = df_calc.groupby("id")["y"].mean().to_numpy()
between_mean = float(np.mean(entity_means))
between_sd = float(np.std(entity_means, ddof=1))
between_min = float(np.min(entity_means))
between_max = float(np.max(entity_means))

# Within: deviations from entity means.
entity_mean_map = df_calc.groupby("id")["y"].mean()
df_calc["entity_mean"] = df_calc["id"].map(entity_mean_map)
y_within = (df_calc["y"] - df_calc["entity_mean"]).to_numpy()
within_mean = float(np.mean(y_within))
within_sd = float(np.std(y_within, ddof=1))
within_min = float(np.min(y_within))
within_max = float(np.max(y_within))

result = {
    "variable": ["lwage"] * 3,
    "type": ["overall", "between", "within"],
    "n": [n_total, n_entities, n_total],
    "mean": [overall_mean, between_mean, within_mean],
    "sd": [overall_sd, between_sd, within_sd],
    "min": [overall_min, between_min, within_min],
    "max": [overall_max, between_max, within_max],
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

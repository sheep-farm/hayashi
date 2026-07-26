# Reference implementation in Python for the Wooldridge wage1 one-sample t-test case.

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

x = df["wage"].dropna().to_numpy()
n = int(len(x))
mu0 = 5.0
mean_x = float(np.mean(x))
sd_x = float(np.std(x, ddof=1))
std_err = float(sd_x / np.sqrt(n))
t_stat = float((mean_x - mu0) / std_err)
df = n - 1
p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
t_crit = float(stats.t.ppf(0.975, df))

result = {
    "test": "one-sample t-test",
    "variable": "wage",
    "n": n,
    "mean": mean_x,
    "mu": mu0,
    "std_err": std_err,
    "t_stat": t_stat,
    "df": df,
    "p_value": p_value,
    "ci_lower": mean_x - t_crit * std_err,
    "ci_upper": mean_x + t_crit * std_err,
}

out_dir = CASE_DIR / "reference"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "expected.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CASE_DIR / "data" / "wagepan.csv"

df = pd.read_csv(DATA_PATH)
y = df["lwage"].values
X = np.column_stack(
    [np.ones(len(df)), df["union"].values, df["married"].values]
)
nr = df["nr"].values

# Pooled OLS residuals
beta = np.linalg.lstsq(X, y, rcond=None)[0]
e = y - X @ beta

entities, counts = np.unique(nr, return_counts=True)
n = len(entities)
Tbar = counts.mean()
N = len(y)

# Hayashi/Greeners Breusch-Pagan LM statistic for individual effects
A = sum((e[nr == i].sum() ** 2) for i in entities) / Tbar
B = float(np.sum(e**2))

lm_stat = (n * Tbar) / (2 * (Tbar - 1)) * ((A / B) - 1) ** 2
p_value = 1 - stats.chi2.cdf(lm_stat, 1)

out = {
    "coefficients": {
        "lm_stat": lm_stat,
        "p_value": p_value,
    },
    "standard_errors": {
        "lm_stat": 0.0,
        "p_value": 0.0,
    },
}

print(json.dumps(out, indent=2))
